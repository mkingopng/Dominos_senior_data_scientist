## Communita SLR

 OLS Regression Results                            
==============================================================================
Dep. Variable:               quantity   R-squared:                       0.699
Model:                            OLS   Adj. R-squared:                  0.696
Method:                 Least Squares   F-statistic:                     193.0
Date:                Sun, 04 Aug 2024   Prob (F-statistic):           2.29e-23
Time:                        13:57:17   Log-Likelihood:                 128.89
No. Observations:                  85   AIC:                            -253.8
Df Residuals:                      83   BIC:                            -248.9
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          3.7463      0.032    118.686      0.000       3.683       3.809
unit_price    -0.0334      0.002    -13.893      0.000      -0.038      -0.029
==============================================================================
Omnibus:                        8.832   Durbin-Watson:                   0.633
Prob(Omnibus):                  0.012   Jarque-Bera (JB):                8.764
Skew:                          -0.637   Prob(JB):                       0.0125
Kurtosis:                       3.923   Cond. No.                         71.4
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

## Communita SLR outliers removed

OLS Regression Results                            
==============================================================================
Dep. Variable:               quantity   R-squared:                       0.694
Model:                            OLS   Adj. R-squared:                  0.690
Method:                 Least Squares   F-statistic:                     183.9
Date:                Sun, 04 Aug 2024   Prob (F-statistic):           1.53e-22
Time:                        14:38:17   Log-Likelihood:                 130.66
No. Observations:                  83   AIC:                            -257.3
Df Residuals:                      81   BIC:                            -252.5
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          3.8237      0.038    101.431      0.000       3.749       3.899
unit_price    -0.0395      0.003    -13.559      0.000      -0.045      -0.034
==============================================================================
Omnibus:                       32.444   Durbin-Watson:                   0.591
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               73.096
Skew:                          -1.376   Prob(JB):                     1.34e-16
Kurtosis:                       6.683   Cond. No.                         88.1
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

## Communita SLR outliers removed and Box-Cox transformed
OLS Regression Results                            
==============================================================================
Dep. Variable:        BoxCox_quantity   R-squared:                       0.686
Model:                            OLS   Adj. R-squared:                  0.682
Method:                 Least Squares   F-statistic:                     177.1
Date:                Sun, 04 Aug 2024   Prob (F-statistic):           4.36e-22
Time:                        16:56:25   Log-Likelihood:                -296.64
No. Observations:                  83   AIC:                             597.3
Df Residuals:                      81   BIC:                             602.1
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
=====================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------
const               249.0144      6.499     38.318      0.000     236.084     261.945
BoxCox_unit_price    -6.7080      0.504    -13.309      0.000      -7.711      -5.705
==============================================================================
Omnibus:                       31.218   Durbin-Watson:                   0.593
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               70.192
Skew:                          -1.318   Prob(JB):                     5.73e-16
Kurtosis:                       6.654   Cond. No.                         87.9
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

## Analysis of Communita SLR models

Here is a detailed comparison and analysis of the three models:

### 1. Model Based on Raw Data

**OLS Regression Results:**

- **R-squared:** 0.699
- **Adj. R-squared:** 0.696
- **F-statistic:** 193.0
- **Prob (F-statistic):** 2.29e-23
- **AIC:** -253.8
- **BIC:** -248.9

**Coefficients:**

- **Intercept (const):** 3.7463 (t = 118.686, p < 0.000)
- **unit_price:** -0.0334 (t = -13.893, p < 0.000)

**Diagnostics:**

- **Omnibus:** 8.832 (p = 0.012)
- **Durbin-Watson:** 0.633
- **Jarque-Bera (JB):** 8.764 (p = 0.0125)
- **Skew:** -0.637
- **Kurtosis:** 3.923
- **Cond. No.:** 71.4

### 2. Model After Removing Outliers

**OLS Regression Results:**

- **R-squared:** 0.694
- **Adj. R-squared:** 0.690
- **F-statistic:** 183.9
- **Prob (F-statistic):** 1.53e-22
- **AIC:** -257.3
- **BIC:** -252.5

**Coefficients:**

- **Intercept (const):** 3.8237 (t = 101.431, p < 0.000)
- **unit_price:** -0.0395 (t = -13.559, p < 0.000)

**Diagnostics:**

- **Omnibus:** 32.444 (p < 0.000)
- **Durbin-Watson:** 0.591
- **Jarque-Bera (JB):** 73.096 (p < 0.000)
- **Skew:** -1.376
- **Kurtosis:** 6.683
- **Cond. No.:** 88.1

### 3. Model After Removing Outliers and Applying Box-Cox Transformation

**OLS Regression Results:**

- **R-squared:** 0.686
- **Adj. R-squared:** 0.682
- **F-statistic:** 177.1
- **Prob (F-statistic):** 4.36e-22
- **AIC:** 597.3
- **BIC:** 602.1

**Coefficients:**

- **Intercept (const):** 249.0144 (t = 38.318, p < 0.000)
- **BoxCox_unit_price:** -6.7080 (t = -13.309, p < 0.000)

**Diagnostics:**

- **Omnibus:** 31.218 (p < 0.000)
- **Durbin-Watson:** 0.593
- **Jarque-Bera (JB):** 70.192 (p < 0.000)
- **Skew:** -1.318
- **Kurtosis:** 6.654
- **Cond. No.:** 87.9

### Analysis and Comparison:

1. **Model Fit (R-squared):**
   - The first model (raw data) has the highest R-squared (0.699), indicating the best fit among the three models. 
   - The second model (after removing outliers) has a slightly lower R-squared (0.694), showing a minor reduction in fit.
   - The third model (after removing outliers and applying Box-Cox transformation) has the lowest R-squared (0.686), indicating a further slight reduction in fit.

2. **Model Coefficients:**
   - The coefficients for `unit_price` are negative across all models, indicating an inverse relationship between price and quantity.
   - The coefficient magnitude increases after removing outliers and applying Box-Cox transformation, suggesting a stronger relationship in the adjusted data.

3. **Statistical Significance:**
   - All models have highly significant F-statistics (p < 0.0001), indicating that the overall regression model is significant.
   - All coefficients are also highly significant (p < 0.0001), indicating strong evidence that `unit_price` is a significant predictor of `quantity`.

4. **Model Diagnostics:**
   - The Omnibus and Jarque-Bera tests for normality show significant p-values for all models, indicating deviation from normality in residuals.
   - Skewness and Kurtosis values indicate the presence of outliers and non-normal distribution of residuals, especially in the second and third models.
   - The Durbin-Watson statistic, which tests for autocorrelation, is consistently below 1 in all models, suggesting positive serial correlation in residuals.

5. **AIC and BIC:**
   - The AIC and BIC values are lowest for the second model (after removing outliers), indicating it may be the preferred model based on these criteria, which balance fit and model complexity.
   - The third model has higher AIC and BIC values due to the transformation and complexity, suggesting it might not provide a better fit despite the normalization.

### Conclusion and Recommendations:

- **Model Selection:** The first model provides the best fit, but the second model (after removing outliers) offers a balance between fit and complexity, with better diagnostic metrics (AIC/BIC). The third model, while normalized, does not significantly improve the fit and increases complexity.
- **Diagnostics and Assumptions:** All models show issues with normality and autocorrelation in residuals. Addressing these through further transformation, or using models that can handle non-linearity and correlation (e.g., Generalized Linear Models or Mixed Effects Models) might be beneficial.
- **Box-Cox Transformation:** While Box-Cox can normalize data, its effectiveness depends on the specific data distribution. Here, it slightly reduced fit, indicating it might not be the best approach for this dataset.
- **Further Steps:** Consider exploring non-linear models, adding more features, or employing regularization techniques to improve model robustness. Validate models on separate data to ensure generalizability.

here is a summary table comparing the key metrics for the three models:

| **Metric**                 | **Model 1 (Raw Data)** | **Model 2 (Outliers Removed)** | **Model 3 (Outliers Removed + Box-Cox)** |
|----------------------------|------------------------|--------------------------------|------------------------------------------|
| **R-squared**              | 0.699                  | 0.694                          | 0.686                                    |
| **Adj. R-squared**         | 0.696                  | 0.690                          | 0.682                                    |
| **F-statistic**            | 193.0                  | 183.9                          | 177.1                                    |
| **Prob (F-statistic)**     | 2.29e-23               | 1.53e-22                       | 4.36e-22                                 |
| **AIC**                    | -253.8                 | -257.3                         | 597.3                                    |
| **BIC**                    | -248.9                 | -252.5                         | 602.1                                    |
| **Intercept (const)**      | 3.7463                 | 3.8237                         | 249.0144                                 |
| **unit_price Coefficient** | -0.0334                | -0.0395                        | -6.7080                                  |
| **unit_price t-value**     | -13.893                | -13.559                        | -13.309                                  |
| **unit_price p-value**     | < 0.0001               | < 0.0001                       | < 0.0001                                 |
| **Omnibus**                | 8.832                  | 32.444                         | 31.218                                   |
| **Prob (Omnibus)**         | 0.012                  | < 0.0001                       | < 0.0001                                 |
| **Durbin-Watson**          | 0.633                  | 0.591                          | 0.593                                    |
| **Jarque-Bera (JB)**       | 8.764                  | 73.096                         | 70.192                                   |
| **Prob (JB)**              | 0.0125                 | < 0.0001                       | < 0.0001                                 |
| **Skew**                   | -0.637                 | -1.376                         | -1.318                                   |
| **Kurtosis**               | 3.923                  | 6.683                          | 6.654                                    |
| **Cond. No.**              | 71.4                   | 88.1                           | 87.9                                     |

### Notes:
- **R-squared and Adj. R-squared:** Indicate how well the independent variable explains the variance in the dependent variable. Higher values represent a better fit.
- **F-statistic and Prob (F-statistic):** Test the overall significance of the model. Higher F-statistic and lower p-values indicate a better fit.
- **AIC and BIC:** Information criteria used for model selection. Lower values indicate a better balance between model fit and complexity.
- **Coefficients:** Represent the impact of `unit_price` on `quantity`. The sign indicates the direction of the relationship.
- **Diagnostics (Omnibus, Durbin-Watson, Jarque-Bera, Skew, Kurtosis):** Indicate potential issues with residuals, such as non-normality, autocorrelation, and outliers. Lower values of these tests indicate fewer issues.

### Summary:
- **Model 1 (Raw Data):** Provides the best fit with the highest R-squared but has issues with non-normality and autocorrelation in residuals.
- **Model 2 (Outliers Removed):** Slightly lower fit than Model 1 but better AIC and BIC values, indicating a more balanced model. However, it still shows significant non-normality and autocorrelation issues.
- **Model 3 (Outliers Removed + Box-Cox):** Further reduction in R-squared but shows an even stronger negative relationship between `unit_price` and `quantity`. The AIC and BIC values are higher due to the transformation and increased complexity, but the diagnostics still indicate issues with non-normality and autocorrelation.

This summary table should help in comparing the performance and characteristics of the three models.

