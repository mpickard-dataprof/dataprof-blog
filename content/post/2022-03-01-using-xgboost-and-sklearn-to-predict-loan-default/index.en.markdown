---
title: Using xgboost and sklearn to Predict Loan Default
author: Matt Pickard
date: '2022-03-01'
slug: using-xgboost-and-sklearn-to-predict-loan-default
categories:
  - python
  - machine learning
tags:
  - python
  - sklearn
  - xgboost
---


```python
import numpy as np 
import pandas as pd 
```

# Introduction
I wanted to combine `xgboost` with `sklearn` pipelines to process a `pandas` DataFrame. Special thanks to Kaggle user M Yasser H for supplying the [Loan Default Dataset](https://www.kaggle.com/yasserh/loan-default-dataset). 

# Load the data

```python
# Load the data
loan_df = pd.read_csv("../data/Loan_Default.csv")
```

# Create labels and features
The loan status (whether or not the customer defaulted on the loan) is the target variable. We'll extract that out as our labels. The other columns (minus the *ID*) will serve as our features. And we'll take a peek at our features.

```python
# Split out labels and features, encode labels as integers
y = loan_df['Status']

X = loan_df.loc[:,~loan_df.columns.isin(['ID','Status'])]
```

# Simple data exploration
Take a peek as our feature variables.

```python
X.head()
```

```
##    year loan_limit             Gender  ... Region Security_Type dtir1
## 0  2019         cf  Sex Not Available  ...  south        direct  45.0
## 1  2019         cf               Male  ...  North        direct   NaN
## 2  2019         cf               Male  ...  south        direct  46.0
## 3  2019         cf               Male  ...  North        direct  42.0
## 4  2019         cf              Joint  ...  North        direct  39.0
## 
## [5 rows x 32 columns]
```

Notice there is a mix of categorical and continuous variables.

Let's check if there are any missing values.

```python
X.isnull().sum()
```

```
## year                             0
## loan_limit                    3344
## Gender                           0
## approv_in_adv                  908
## loan_type                        0
## loan_purpose                   134
## Credit_Worthiness                0
## open_credit                      0
## business_or_commercial           0
## loan_amount                      0
## rate_of_interest             36439
## Interest_rate_spread         36639
## Upfront_charges              39642
## term                            41
## Neg_ammortization              121
## interest_only                    0
## lump_sum_payment                 0
## property_value               15098
## construction_type                0
## occupancy_type                   0
## Secured_by                       0
## total_units                      0
## income                        9150
## credit_type                      0
## Credit_Score                     0
## co-applicant_credit_type         0
## age                            200
## submission_of_application      200
## LTV                          15098
## Region                           0
## Security_Type                    0
## dtir1                        24121
## dtype: int64
```


# Preprocessing

Since we have a mix of continuous and categorical variables, we'll setup an imputers for each type of variable. So, we are going to separate teh continous and the categorical varabiles into separate DataFrames. 

For the continuous variables, we'll impute the median.

```python
from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer

# extract numeric columns
numeric_mask = (X.dtypes != object)
numeric_columns = X.columns[numeric_mask].tolist()
numeric_df = X[numeric_columns]

# create "imputer", just going to fill missing values with "missing"
numeric_imputor = DataFrameMapper(
  [([numeric_feature], SimpleImputer(strategy='median')) for numeric_feature in numeric_df],
  input_df=True,
  df_out=True
  )
```

For the categorical variables, we'll impute the value 'missing'.

```python
# extract categorical features
categorical_mask = (X.dtypes == object)
categorical_columns = X.columns[categorical_mask].tolist()
categorical_df = X[categorical_columns]

categorical_imputor = DataFrameMapper(
  [([categorical_feature], SimpleImputer(strategy='constant', fill_value = "missing")) for categorical_feature in categorical_df],
  input_df=True,
  df_out=True
  )
```

# Build the pipeline

We are going to use `sklearn`'s `DictVectorizer`, which operates on numpy arrays/matrices. So to make it compatiable with DataFrames, we'll create a simple utility class to allow a DataFrame to be passed through the pipeline. Thanks to Chanseok for the [dictifier code](https://goodboychan.github.io/python/datacamp/machine_learning/2020/07/07/03-Using-XGBoost-in-pipelines.html).

```python
from sklearn.base import BaseEstimator, TransformerMixin

# Define Dictifier class to turn df into dictionary as part of pipeline
class Dictifier(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  
  def transform(self, X):
    if type(X) == pd.core.frame.DataFrame:
      return X.to_dict("records")
    else:
      return pd.DataFrame(X).to_dict("records")
```

Now we build the pipeline.  Notice how we use the FeatureUnion to bring the continous and categorical features back together again at the start of the pipeline.

```python
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb

imputed_df = FeatureUnion([
  ('num_imputer', numeric_imputor),
  ('cat_imputer', categorical_imputor)    
  ])
  
xgb_pipeline = Pipeline([
  ("featureunion", imputed_df),
  ('dictifier', Dictifier()),
  ('dict_vectorizer', DictVectorizer(sort=False)),
  ("xgb", xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
  ])
```

# 3-Fold Cross Validation
We'll use 3-fold cross validation (instead of 10, or something greater) to minimize compute time.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(xgb_pipeline, X, y, scoring="f1", cv=3)
avg_f1 = np.mean(np.sqrt(np.abs(scores)))
print("Avg F1 Score:", avg_f1)
```

```
## Avg F1 Score: 0.9999863537583441
```

# Conclusions
The average F1 score is suspiciously high, so let's not put much clout in the quality of the model. But it serves the purpose of demonstrating how to pass a 'pandas' DataFrame through an `sklearn` pipeline, preprocess mixed variable (continuous and categorical) data, and build an xgboost classifier.

Just because it's bugging me, here are a few things that may need to be improved in this model:

1) There is probably a high correlation between the target variable and some feature variables. We can check this quickly. Fille the NAs with zero and correlate it with the loan status (which is 0 and 1).


```python
numeric_fillna_df = numeric_df.fillna(0)
numeric_fillna_df.corrwith(y)
```

```
## year                         NaN
## loan_amount            -0.036825
## rate_of_interest       -0.958875
## Interest_rate_spread   -0.392977
## Upfront_charges        -0.431183
## term                   -0.000675
## property_value         -0.273267
## income                 -0.044620
## Credit_Score            0.004004
## LTV                    -0.267700
## dtir1                  -0.325613
## dtype: float64
```
We can see that *rate_of_interest* has a high inverse correlation with the target variable.  However, I did try removing *rate_of_interest* and still ended up with an F1 score of 0.9999.

2) Find a better way to impute the missing categorical. Chirag Goyal enumerates some options in [this post](https://www.analyticsvidhya.com/blog/2021/04/how-to-handle-missing-values-of-categorical-variables/). I suspect that building a model to predict missing values would be an option.  Another simpler option would be to just randomly insert existing values.  But, currently, with the imputer in this post, it is essentially treating 'missing' as a legit value.
