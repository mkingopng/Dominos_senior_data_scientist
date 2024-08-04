import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numdifftools as nd
from scipy.stats import skew, kurtosis, boxcox
from scipy.special import inv_boxcox

# Load data
df = pd.read_csv('./../data/price_elasticity_data.csv')

# Rename columns
names = {'Menu_Price': 'unit_price', 'Pizza_Count': 'quantity', 'Profit_Percentage': 'profit_percentage'}
df.rename(columns=names, inplace=True)

# Calculate additional features
df['gross_margin'] = df['profit_percentage'] / 100
df['gross_profit'] = df['unit_price'] * (1 - df['gross_margin'])
df['cost'] = df['unit_price'] - df['gross_profit']

# Subset the dataframe for 'Communita' product and create a copy
communita_df = df[df['Product'] == 'Communita'].copy()

# Outlier detection and removal
def identify_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)], lower_bound, upper_bound

communita_outliers, comm_lower, comm_upper = identify_outliers(communita_df, 'unit_price')
data_no_outliers = communita_df.drop(communita_outliers[communita_outliers['unit_price'] > comm_upper].index)
communita_df = data_no_outliers.drop(communita_outliers[communita_outliers['unit_price'] < comm_lower].index)

# Apply Box-Cox transformation
def apply_boxcox_and_calculate_stats(df, features):
    stats = {}
    lambdas = {}
    for feature in features:
        df[f'BoxCox_{feature}'], fitted_lambda = boxcox(df[feature] + 1)
        lambdas[feature] = fitted_lambda
        skewness = skew(df[f'BoxCox_{feature}'])
        kurt = kurtosis(df[f'BoxCox_{feature}'])
        stats[feature] = (skewness, kurt)
        print(f"Feature: {feature}")
        print(f"Box-Cox Transformed: Skewness = {skewness:.4f}, Kurtosis = {kurt:.4f}\n")
    return df, stats, lambdas

features = ['unit_price', 'quantity', 'gross_profit', 'cost']
communita_df, communita_stats, fitted_lambdas = apply_boxcox_and_calculate_stats(communita_df, features)

# Calculate mean cost from the original data
mean_cost = np.mean(communita_df['cost'])

# Fit linear regression model
x_value = communita_df['BoxCox_unit_price']
y_value = communita_df['BoxCox_quantity']

# Ensure x and y are of the same length
assert len(x_value) == len(y_value), "x and y must have the same length"

# Add a constant to the independent variable (unit_price)
X = add_constant(x_value)

# Fit the regression model
model = sm.OLS(y_value, X)
communita_result = model.fit()

# Print the summary of the regression model
print(communita_result.summary())

# Get the predicted values in Box-Cox transformed space
predictions = communita_result.predict(X)

# Inverse transform the predictions to the original scale
predictions_original = inv_boxcox(predictions, fitted_lambdas['quantity'])

# Add predictions to the dataframe
communita_df['predicted_quantities'] = predictions_original

# Create a scatter plot with the regression line based on the original data
plt.figure(figsize=(10, 6))
plt.scatter(
    communita_df['unit_price'],
    communita_df['quantity'],
    alpha=0.5,
    label='Data points'
)
plt.plot(
    communita_df['unit_price'],
    predictions_original,
    color='red',
    linewidth=2,
    label='Regression line'
)
plt.title('Scatter plot of Unit Price vs Quantity with Regression Line')
plt.xlabel('Unit Price')
plt.ylabel('Quantity')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the gross profit based on the inverse transformed predictions
communita_df['gross_profit_predictions'] = (communita_df['unit_price'] - communita_df['cost']) * communita_df['predicted_quantities']

# Find the price that maximizes gross profit
optimal_price = communita_df.loc[communita_df['gross_profit_predictions'].idxmax(), 'unit_price']
max_gross_profit = communita_df['gross_profit_predictions'].max()
optimal_quantity = communita_df.loc[communita_df['gross_profit_predictions'].idxmax(), 'predicted_quantities']

# Define the gross profit function based on the regression model in the original space
def gross_profit_function(price):
    # Box-Cox transform price
    boxcox_price = (price + 1) ** fitted_lambdas['unit_price'] - 1
    return (price - mean_cost) * (communita_result.params['const'] + communita_result.params['BoxCox_unit_price'] * boxcox_price)

# Calculate the first derivative (slope) at the optimal price
derivative_function = nd.Derivative(gross_profit_function)
slope_at_optimal_price = derivative_function(optimal_price)

# Generate x values for the tangent line
x_vals = np.linspace(communita_df['unit_price'].min(), communita_df['unit_price'].max(), 100)

# Calculate y values using the line equation
y_vals = slope_at_optimal_price * (x_vals - optimal_price) + max_gross_profit

# Plot the original gross profit curve
plt.figure(figsize=(10, 6))
plt.plot(
    communita_df['unit_price'],
    communita_df['gross_profit_predictions'],
    label='Gross Profit Curve'
)
plt.scatter(
    optimal_price,
    max_gross_profit,
    color='red',
    label=f'Optimal Price: {optimal_price:.2f}'
)
plt.plot(
    x_vals,
    y_vals,
    color='red',
    linestyle='--',
    label=f'Tangent Line at Optimal Price'
)
plt.title('Gross Profit Curve (Original Data)')
plt.xlabel('Unit Price')
plt.ylabel('Gross Profit')
plt.legend()
plt.grid(True)
plt.show()

# Print the optimal price and slope at that point
print(f'Optimal Price (Original): {round(optimal_price, 2)}')
print(f'Optimal Quantity (Original): {round(optimal_quantity, 0)}')
print(f'Maximum Gross Profit: {round(max_gross_profit, 2)}')
print(f'Slope of the Gross Profit Function at Optimal Price: {slope_at_optimal_price}') # fix_me
