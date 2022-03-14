---
title: regex to Preserve Currency Tokens
author: Matt Pickard
date: '2022-03-14'
slug: regex-to-preserve-currency-tokens
categories:
  - regex
  - data preparation
tags:
  - python
  - r
  - regex
---

# Introduction
I've been working to collect a corpus of tax legislation, regulations, and court cases. At the core of my corpus is the United States Code Title 26 (USC26), otherwise known as the Internal Revenue Code or the U.S. Tax Code. I'm using the corpus to train a tax-specific word embedding. Word embeddings capture similarity semantics of tokens (words, or multi-word phrases) by encoding them into a n-dimensional space. Similar tokens are embedded in in the n-dimensional space in close proximity.

Since currency values may have important semantic meaning in the USC--for instance, tax bracket boundaries--I want to preserve them so the tokenizer does not break them apart. To do this, I want to replace spaces, commas, decimals, and dollar signs with underscores.  Here are some examples:

| Original     | Preserved       |
|--------------|-----------------|
| $172,175     | 172_175_usd     |
| $2.5 million | 2_5_million_usd |
| $56,750.25   | 56_750_25_usd   |


# Implementation
The approach I took was to first match the different currency formats and then pass the match to a helper function that would eplace spaces, commas, decimals, and dollar signs with underscores.

Here is the pattern I created to match currency tokens:


```python
pattern = r'\$\d{0,3}(\,\d{3}){0,4}(\.\d{1,2})?( (million|billion))?'
```

* `\$` matches a dollar sign. 
* `\d{0,3}` The number after the dollar sign and before a comma or decimal boundary. Looking for 0 to 3 digits allows for these patterns: \$.75, \$125,000, etc.
* `(\,\d{3}){0,4}(\.\d{1,2})?` matches repeated comma boundaries (if they exist) and two-decimal places for cents (if they exist). More specifically:
  * `(\,\d{3})` matches ",xxx". With the addition of `{0,4}` it matches ",xxx" repeatedly--up to  trillions (which is probably overkill for the USC26 because most large currency values has "million" or "billion" text suffixes).
  * `(\.\d{1,2})?` matches an optional cents.
* `( (million|billion))?` just checks for a text suffix (the `?` at the end makes it optional.)

Next, here is a simple helper function to replace spaces, commas, decimals, and dollar signs with underscores and tack on "_usd" on the end. Python's string `replace()` method is faster than the regex `sub()`, so I went with `replace()`.


```python
import re

def replace_currency_parts(match):
    text = match.group(0).replace(" ", "_").replace(".", "_").replace(',', '_')
    if (text[0] == '$'):
        text = text[1:] + "_usd"
    return text
```

Now I need a wishful test string.


```python
test_string = "I made $755.34 billion this year, $5.34 million last year, but I also want to match $125,234.34 and $1,342.40 and $45.09 and $45 in case that's a more realistic salary in my life."
```

Then I compile the regex and replace matches.  Notice that the second argument of `sub()` can take a method that returns a string.  I leverage this to call the helper function.


```python
compiled_pattern = re.compile(pattern)
compiled_pattern.sub(replace_currency_parts, test_string)
```

```
## "I made 755_34_billion_usd this year, 5_34_million_usd last year, but I also want to match 125_234_34_usd and 1_342_40_usd and 45_09_usd and 45_usd in case that's a more realistic salary in my life."
```

Here is the full Python implementation.


```python
import time
import timeit

pattern = r'\$\d{0,3}(\,\d{3}){0,4}(\.\d{1,2})?( (million|billion))?'

def replace_currency_parts(match):
    text = match.group(0).replace(" ", "_").replace(".", "_").replace(',', '_')
    if (text[0] == '$'):
        text = text[1:] + "_usd"
    return text

test_string = "I made $355.34 million this year, $435.34 billion last year, but I also want to match $125,234.34 and $1,342.40 and $45.09 and $45."

compiled_pattern = re.compile(pattern)
compiled_pattern.sub(replace_currency_parts, test_string)
```



# R Implementation
I was curious if python was that much faster than R.  So, here is an R implementation. Like Python's `sub()`, the `str_replace_all` function can take a helper function.


```r
library(stringr)

pattern <- '\\$\\d{0,3}(\\,\\d{3}){0,4}(\\.\\d{1,2})?( (million|billion))?'

replace_currency_parts <- function(match) {
  
  text <- match %>% 
    str_replace_all(" ", "_") %>% 
    str_replace_all("\\.", "_") %>% 
    str_replace_all("\\,", "_")
    
  if (str_sub(text, 1, 1) == "$"){
    text <- str_c(str_sub(text, 2), "_usd")
  }
    
  return(text)
}

test_string <-  "I made $355.34 million this year, $435.34 billion last year, but I also want to match $125,234.34 and $1,342.40 and $45.09 and $45."

str_replace_all(test_string, pattern, replace_currency_parts)
```

```
## [1] "I made 355_34_million_usd this year, 435_34_billion_usd last year, but I also want to match 125_234_34_usd and 1_342_40_usd and 45_09_usd and 45_usd."
```

# Which is faster?
So, which is faster?

I compare the Python and R implementations by taking the average of 1000 executions of the code.  Since the {stringr} package in R is higher level--there's no option to compile the regex ahead of time and keep it out of the loop--I placed the Python regex compile code inside the loop to make a more fair comparison.  In both cases, the units are milliseconds. NOTE: The python `time()` method returns seconds, so I convert.


```python
from time import time

start_ms = int(round(time() * 1000))
for i in range(1000):
  compiled_pattern = re.compile(pattern)
  str = compiled_pattern.sub(replace_currency_parts, test_string)
end_ms = int(round(time() * 1000))

print("elapsed ms = ", (end_ms-start_ms)/1000)
```

```
## elapsed ms =  0.019
```


```r
library(microbenchmark)

microbenchmark::microbenchmark(str_replace_all(test_string, pattern, replace_currency_parts), times = 1000)
```

```
## Unit: milliseconds
##                                                           expr    min      lq
##  str_replace_all(test_string, pattern, replace_currency_parts) 1.7242 1.77735
##      mean median      uq    max neval
##  1.821252 1.7906 1.80905 3.5674  1000
```

I don't know what overhead is included in the {stringr} package, but it's about two orders of magnitude slower than Python. I suspect its Python's string `replace()` method which is really speedy.
