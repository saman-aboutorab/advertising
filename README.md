# Advertising campaign sales
:::

::: {#2e25fdd8-4d84-45bc-80f0-949917e00a17 .cell .code collapsed="false" jupyter="{\"outputs_hidden\":false,\"source_hidden\":true}"}
``` python
# Importing pandas
import pandas as pd

# Importing the course datasets 
diabetes = pd.read_csv('datasets/diabetes_clean.csv')
music = pd.read_csv('datasets/music_clean.csv')
advertising = pd.read_csv('datasets/advertising_and_sales_clean.csv')
telecom = pd.read_csv("datasets/telecom_churn_clean.csv")
```
:::

::: {#0e7949e8 .cell .markdown}
## Introduction

In this project, we will work with a dataset called sales_df, which
contains information on advertising campaign expenditure across
different media types, and the number of dollars generated in sales for
the respective campaign.

![image](vertopal_0a30be8736874f91b57f0d8fe33a44ab/6a73857a2b84963ad312512f67458dfcc4cfcbea.jpg)

Source:
<https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn>
:::

::: {#079c7628-464c-471b-9867-b684acc232b9 .cell .code execution_count="34" executionCancelledAt="null" executionTime="48" lastExecutedAt="1687449385066" lastScheduledRunId="null" lastSuccessfullyExecutedCode="# Import the necessary modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso"}
``` python
# Import the necessary modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
```
:::

::: {#60d74eed-e13f-4433-baa9-83b75eae1883 .cell .code execution_count="21" executionCancelledAt="null" executionTime="21" lastExecutedAt="1687448946849" lastScheduledRunId="null" lastSuccessfullyExecutedCode="sales_df = pd.read_csv('advertising_and_sales_clean.csv')
print(sales_df.head(5))"}
``` python
sales_df = pd.read_csv('advertising_and_sales_clean.csv')
print(sales_df.head(5))
```

::: {.output .stream .stdout}
            tv     radio  social_media influencer      sales
    0  16000.0   6566.23       2907.98       Mega   54732.76
    1  13000.0   9237.76       2409.57       Mega   46677.90
    2  41000.0  15886.45       2913.41       Mega  150177.83
    3  83000.0  30020.03       6922.30       Mega  298246.34
    4  15000.0   8437.41       1406.00      Micro   56594.18
:::
:::

::: {#8596ba0a-97d0-4e53-9f8e-5ae412bcd1cd .cell .markdown}
## Creating features
:::

::: {#71cf7ce3-d98d-4839-a850-5fcb3658498c .cell .code execution_count="15" executionCancelledAt="null" executionTime="13" lastExecutedAt="1687448822096" lastScheduledRunId="null" lastSuccessfullyExecutedCode="# Create X from the radio column's values
X = sales_df['radio'].values

# Create y from the sales column's values
y = sales_df['sales'].values

# Reshape X
X = X.reshape(-1, 1)

# Check the shape of the features and targets
print(X.shape, y.shape)"}
``` python
# Create X from the radio column's values
X = sales_df['radio'].values

# Create y from the sales column's values
y = sales_df['sales'].values

# Reshape X
X = X.reshape(-1, 1)

# Check the shape of the features and targets
print(X.shape, y.shape)
```

::: {.output .stream .stdout}
    (4546, 1) (4546,)
:::
:::

::: {#3573cb22-d12f-4426-ab80-0e6471631ffa .cell .markdown}
## Building a linear regression model
:::

::: {#9d316389-e165-44bb-b0d3-d86693386cac .cell .code execution_count="16" executionCancelledAt="null" executionTime="13" lastExecutedAt="1687448822442" lastScheduledRunId="null" lastSuccessfullyExecutedCode="# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)

print(predictions[:5])"}
``` python
# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)

print(predictions[:5])
```

::: {.output .stream .stdout}
    [ 95491.17119147 117829.51038393 173423.38071499 291603.11444202
     111137.28167129]
:::
:::

::: {#80a4f2ea-e646-40e3-b99a-1f5f6b8e41a1 .cell .markdown}
## Visualizing a linear regression model
:::

::: {#324e035c-8cb4-42b5-a03f-fc3fa40bd8e6 .cell .code execution_count="17" executionCancelledAt="null" executionTime="152" lastExecutedAt="1687448822940" lastScheduledRunId="null" lastSuccessfullyExecutedCode="# Create scatter plot
plt.scatter(X, y, color=\"blue\")

# Create line plot
plt.plot(X, predictions, color=\"red\")
plt.xlabel(\"Radio Expenditure ($)\")
plt.ylabel(\"Sales ($)\")

# Display the plot
plt.show()"}
``` python
# Create scatter plot
plt.scatter(X, y, color="blue")

# Create line plot
plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()
```

::: {.output .display_data}
![](vertopal_0a30be8736874f91b57f0d8fe33a44ab/186e808e62cad01209e2971ce959a589a0139423.png)
:::
:::

::: {#0fb55db0-37d2-491f-a6e4-d204e3b8354d .cell .markdown}
## Fit and predict for regression
:::

::: {#7447aa4e-ab44-4ae6-9ba9-74e5c111060b .cell .code execution_count="22" executionCancelledAt="null" executionTime="54" lastExecutedAt="1687449026254" lastScheduledRunId="null" lastSuccessfullyExecutedCode="# Create X and y arrays
X = sales_df.drop([\"sales\",\"influencer\"], axis=1).values
y = sales_df[\"sales\"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)
print(\"Predictions: {}, Actual Values: {}\".format(y_pred[:2], y_test[:2]))"}
``` python
# Create X and y arrays
X = sales_df.drop(["sales","influencer"], axis=1).values
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))
```

::: {.output .stream .stdout}
    Predictions: [53176.66154234 70996.19873235], Actual Values: [55261.28 67574.9 ]
:::
:::

::: {#15e64476-8c42-4c0a-9fcf-6f0815016993 .cell .markdown}
## Regression performance
:::

::: {#8d5a3c34-32f7-4a70-a91d-7c31af18eaf4 .cell .code execution_count="24" executionCancelledAt="null" executionTime="49" lastExecutedAt="1687449081667" lastScheduledRunId="null" lastSuccessfullyExecutedCode="# Compute R-squared
r_squared = reg.score(X_test, y_test)

# Compute RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the metrics
print(\"R^2: {}\".format(r_squared))
print(\"RMSE: {}\".format(rmse))"}
``` python
# Compute R-squared
r_squared = reg.score(X_test, y_test)

# Compute RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))
```

::: {.output .stream .stdout}
    R^2: 0.9990152104759368
    RMSE: 2944.4331996001006
:::
:::

::: {#d65536d2-323e-4903-911a-8ef4fce5d710 .cell .markdown}
## Cross-validation for R-squared
:::

::: {#79a5163c-eb5b-471b-94df-3445dcd1b7cd .cell .code execution_count="30" executionCancelledAt="null" executionTime="46" lastExecutedAt="1687449306638" lastScheduledRunId="null" lastSuccessfullyExecutedCode="# Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_results = cross_val_score(reg, X, y, cv=kf)

# Print scores
print(cv_results)"}
``` python
# Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_results = cross_val_score(reg, X, y, cv=kf)

# Print scores
print(cv_results)
```

::: {.output .stream .stdout}
    [0.99894062 0.99909245 0.9990103  0.99896344 0.99889153 0.99903953]
:::
:::

::: {#4f9c24ae-4ac1-4f98-b025-4f0578408f22 .cell .code execution_count="31" executionCancelledAt="null" executionTime="12" lastExecutedAt="1687449307233" lastScheduledRunId="null" lastSuccessfullyExecutedCode="# Print the mean
print(np.mean(cv_results))

# Print the standard deviation
print(np.std(cv_results))

# Print the 95% confidence interval
print(np.quantile(cv_results, [0.025, 0.975]))"}
``` python
# Print the mean
print(np.mean(cv_results))

# Print the standard deviation
print(np.std(cv_results))

# Print the 95% confidence interval
print(np.quantile(cv_results, [0.025, 0.975]))
```

::: {.output .stream .stdout}
    0.9989896443678249
    6.608118371529651e-05
    [0.99889767 0.99908583]
:::
:::

::: {#1fbd3a88-c243-46e0-bbc1-2cdaff94aa31 .cell .markdown}
## Regularized regression: Ridge
:::

::: {#ccc850a4-df32-4af6-bc58-881db2d423b0 .cell .code execution_count="33" executionCancelledAt="null" executionTime="95" lastExecutedAt="1687449357388" lastScheduledRunId="null" lastSuccessfullyExecutedCode="alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []
for alpha in alphas:
  
  # Create a Ridge regression model
  ridge = Ridge(alpha=alpha)
  
  # Fit the data
  ridge.fit(X_train, y_train)
  
  # Obtain R-squared
  score = ridge.score(X_test, y_test)
  ridge_scores.append(score)
print(ridge_scores)"}
``` python
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []
for alpha in alphas:
  
  # Create a Ridge regression model
  ridge = Ridge(alpha=alpha)
  
  # Fit the data
  ridge.fit(X_train, y_train)
  
  # Obtain R-squared
  score = ridge.score(X_test, y_test)
  ridge_scores.append(score)
print(ridge_scores)
```

::: {.output .stream .stdout}
    [0.9990152104759369, 0.9990152104759373, 0.9990152104759419, 0.9990152104759871, 0.9990152104764387, 0.9990152104809561]
:::
:::

::: {#9d8ec469-77cf-4ef8-a573-56cccc994bcd .cell .markdown}
## Lasso regression for feature importance
:::

::: {#8cec7d5f-55d7-4425-b41c-8c697e767339 .cell .code execution_count="37" executionCancelledAt="null" executionTime="267" lastExecutedAt="1687449523096" lastScheduledRunId="null" lastSuccessfullyExecutedCode="# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Fit the model to the data
lasso.fit(X, y)

# Compute and print the coefficients
sales_columns = ['tv', 'radio', 'social_media']
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()"}
``` python
# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Fit the model to the data
lasso.fit(X, y)

# Compute and print the coefficients
sales_columns = ['tv', 'radio', 'social_media']
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()
```

::: {.output .stream .stdout}
    [ 3.56256962 -0.00397035  0.00496385]
:::

::: {.output .display_data}
![](vertopal_0a30be8736874f91b57f0d8fe33a44ab/aad6a51a690f9a4771f48718763d0851fd6d3eb9.png)
:::
:::
