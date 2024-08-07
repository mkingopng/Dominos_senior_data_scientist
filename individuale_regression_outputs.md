## individuale SLR

 OLS Regression Results                            
==============================================================================
Dep. Variable:               quantity   R-squared:                       0.612
Model:                            OLS   Adj. R-squared:                  0.606
Method:                 Least Squares   F-statistic:                     108.6
Date:                Sun, 04 Aug 2024   Prob (F-statistic):           8.21e-16
Time:                        14:18:36   Log-Likelihood:                -3.0430
No. Observations:                  71   AIC:                             10.09
Df Residuals:                      69   BIC:                             14.61
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          6.6425      0.147     45.114      0.000       6.349       6.936
unit_price    -0.1639      0.016    -10.423      0.000      -0.195      -0.133
==============================================================================
Omnibus:                       50.515   Durbin-Watson:                   0.214
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              162.982
Skew:                           2.320   Prob(JB):                     4.06e-36
Kurtosis:                       8.793   Cond. No.                         45.8
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

## individuale SLR outliers removed

OLS Regression Results                            
==============================================================================
Dep. Variable:               quantity   R-squared:                       0.613
Model:                            OLS   Adj. R-squared:                  0.607
Method:                 Least Squares   F-statistic:                     106.0
Date:                Sun, 04 Aug 2024   Prob (F-statistic):           1.95e-15
Time:                        14:58:15   Log-Likelihood:                 5.7431
No. Observations:                  69   AIC:                            -7.486
Df Residuals:                      67   BIC:                            -3.018
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          6.8196      0.166     41.050      0.000       6.488       7.151
unit_price    -0.1856      0.018    -10.295      0.000      -0.222      -0.150
==============================================================================
Omnibus:                       48.307   Durbin-Watson:                   0.179
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              174.110
Skew:                           2.151   Prob(JB):                     1.56e-38
Kurtosis:                       9.485   Cond. No.                         56.9
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

## individuale SLR outliers removed and Box-Cox transformed

OLS Regression Results                            
==============================================================================
Dep. Variable:        BoxCox_quantity   R-squared:                       0.817
Model:                            OLS   Adj. R-squared:                  0.815
Method:                 Least Squares   F-statistic:                     300.0
Date:                Sun, 04 Aug 2024   Prob (F-statistic):           1.94e-26
Time:                        19:49:20   Log-Likelihood:                 1722.9
No. Observations:                  69   AIC:                            -3442.
Df Residuals:                      67   BIC:                            -3437.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
=====================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------
const                 0.0816   2.18e-12   3.74e+10      0.000       0.082       0.082
BoxCox_unit_price  -2.68e-12   1.55e-13    -17.320      0.000   -2.99e-12   -2.37e-12
==============================================================================
Omnibus:                        1.695   Durbin-Watson:                   0.590
Prob(Omnibus):                  0.429   Jarque-Bera (JB):                1.226
Skew:                          -0.017   Prob(JB):                        0.542
Kurtosis:                       2.348   Cond. No.                         73.2
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

## Analysis of individuale SLR models

### Summary of Model Outputs:

#### **Model 1: Individuale SLR (Raw Data)**

- **R-squared:** 0.612
- **Adj. R-squared:** 0.606
- **F-statistic:** 108.6
- **Prob (F-statistic):** 8.21e-16
- **AIC:** 10.09
- **BIC:** 14.61
- **Intercept (const):** 6.6425
- **unit_price Coefficient:** -0.1639
- **Omnibus:** 50.515
- **Prob (Omnibus):** 0.000
- **Durbin-Watson:** 0.214
- **Jarque-Bera (JB):** 162.982
- **Prob (JB):** 4.06e-36
- **Skew:** 2.320
- **Kurtosis:** 8.793
- **Cond. No.:** 45.8

#### **Model 2: Individuale SLR (Outliers Removed)**

- **R-squared:** 0.613
- **Adj. R-squared:** 0.607
- **F-statistic:** 106.0
- **Prob (F-statistic):** 1.95e-15
- **AIC:** -7.486
- **BIC:** -3.018
- **Intercept (const):** 6.8196
- **unit_price Coefficient:** -0.1856
- **Omnibus:** 48.307
- **Prob (Omnibus):** 0.000
- **Durbin-Watson:** 0.179
- **Jarque-Bera (JB):** 174.110
- **Prob (JB):** 1.56e-38
- **Skew:** 2.151
- **Kurtosis:** 9.485
- **Cond. No.:** 56.9

#### **Model 3: Individuale SLR (Outliers Removed + Box-Cox Transformed)**

- **R-squared:** 0.817
- **Adj. R-squared:** 0.815
- **F-statistic:** 300.0
- **Prob (F-statistic):** 1.94e-26
- **AIC:** -3442
- **BIC:** -3437
- **Intercept (const):** 0.0816
- **BoxCox_unit_price Coefficient:** -2.68e-12
- **Omnibus:** 1.695
- **Prob (Omnibus):** 0.429
- **Durbin-Watson:** 0.590
- **Jarque-Bera (JB):** 1.226
- **Prob (JB):** 0.542
- **Skew:** -0.017
- **Kurtosis:** 2.348
- **Cond. No.:** 73.2

### Analysis and Comparison:

1. **Model Fit:**
   - **Model 3** has the highest R-squared (0.817) and Adj. R-squared (0.815), indicating that it explains the variance in the dependent variable better than the other two models.
   - **Model 1** and **Model 2** have similar R-squared and Adj. R-squared values, with **Model 2** performing slightly better due to the removal of outliers.

2. **Statistical Significance:**
   - All models show significant F-statistics with p-values close to 0, indicating that the models are statistically significant.
   - The coefficients for `unit_price` in all models are statistically significant with p-values < 0.0001.

3. **Model Performance (AIC and BIC):**
   - **Model 3** has the lowest AIC and BIC values, suggesting that it provides the best balance between model fit and complexity.
   - **Model 2** has a lower AIC and BIC compared to **Model 1**, indicating an improvement after removing outliers.

4. **Residual Diagnostics:**
   - **Model 3** shows the best residual diagnostics with the lowest Omnibus, Jarque-Bera, and Skew values, indicating that the residuals are more normally distributed.
   - **Model 1** and **Model 2** have higher Omnibus and Jarque-Bera statistics, indicating potential issues with non-normality in residuals.

5. **Durbin-Watson Statistic:**
   - All models have Durbin-Watson statistics significantly below 2, indicating potential positive autocorrelation in residuals.

### Summary:
- **Model 3** (with outliers removed and Box-Cox transformation applied) is the best-performing model in terms of R-squared, AIC, and BIC values, and shows the best residual diagnostics.
- **Model 2** (with outliers removed) provides a slight improvement over **Model 1** (raw data) but still shows significant issues with residuals.
- Based on this analysis, **Model 3** is recommended for further analysis and potential implementation, as it provides the best balance between model fit, complexity, and residual diagnostics.

Here is a table summarising the results for the three models:

| **Metric**                   | **Model 1 (Raw Data)** | **Model 2 (Outliers Removed)** | **Model 3 (Outliers Removed + Box-Cox)** |
|------------------------------|------------------------|--------------------------------|------------------------------------------|
| **R-squared**                | 0.612                  | 0.613                          | 0.817                                    |
| **Adj. R-squared**           | 0.606                  | 0.607                          | 0.815                                    |
| **F-statistic**              | 108.6                  | 106.0                          | 300.0                                    |
| **Prob (F-statistic)**       | 8.21e-16               | 1.95e-15                       | 1.94e-26                                 |
| **AIC**                      | 10.09                  | -7.486                         | -3442                                    |
| **BIC**                      | 14.61                  | -3.018                         | -3437                                    |
| **Intercept (const)**        | 6.6425                 | 6.8196                         | 0.0816                                   |
| **Coefficient (unit_price)** | -0.1639                | -0.1856                        | -2.68e-12                                |
| **t-value (unit_price)**     | -10.423                | -10.295                        | -17.320                                  |
| **p-value (unit_price)**     | < 0.0001               | < 0.0001                       | < 0.0001                                 |
| **Omnibus**                  | 50.515                 | 48.307                         | 1.695                                    |
| **Prob (Omnibus)**           | 0.000                  | 0.000                          | 0.429                                    |
| **Durbin-Watson**            | 0.214                  | 0.179                          | 0.590                                    |
| **Jarque-Bera (JB)**         | 162.982                | 174.110                        | 1.226                                    |
| **Prob (JB)**                | 4.06e-36               | 1.56e-38                       | 0.542                                    |
| **Skew**                     | 2.320                  | 2.151                          | -0.017                                   |
| **Kurtosis**                 | 8.793                  | 9.485                          | 2.348                                    |
| **Cond. No.**                | 45.8                   | 56.9                           | 73.2                                     |

### Summary of the Key Metrics:

1. **Model Fit (R-squared and Adj. R-squared):**
   - **Model 3** shows the highest R-squared and Adj. R-squared values, indicating the best fit to the data.
   - **Model 1** and **Model 2** have similar R-squared and Adj. R-squared values, with **Model 2** being slightly better.

2. **Statistical Significance (F-statistic and p-value):**
   - All models are statistically significant with very low p-values for the F-statistic.
   - The coefficients for `unit_price` are also statistically significant in all models with p-values < 0.0001.

3. **Model Performance (AIC and BIC):**
   - **Model 3** has the lowest AIC and BIC values, suggesting it is the best model considering both fit and complexity.
   - **Model 2** has improved AIC and BIC values compared to **Model 1** after removing outliers.

4. **Residual Diagnostics (Omnibus, Jarque-Bera, Skew, Kurtosis):**
   - **Model 3** exhibits the best residual diagnostics with the lowest Omnibus and Jarque-Bera statistics, indicating better normality of residuals.
   - **Model 1** and **Model 2** have significant issues with non-normality of residuals.

5. **Durbin-Watson Statistic:**
   - All models have Durbin-Watson statistics significantly below 2, indicating potential positive autocorrelation in residuals.

### Conclusion:
- **Model 3** (with outliers removed and Box-Cox transformation applied) is the best-performing model in terms of R-squared, AIC, BIC, and residual diagnostics.
- **Model 2** shows a slight improvement over **Model 1** after removing outliers.
- **Model 3** is recommended for further analysis and potential implementation due to its superior fit and better residual diagnostics.
