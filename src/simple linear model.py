import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Read the data
df = pd.read_csv('./../data/price_elasticity_data.csv')

# Rename columns
names = {'Menu_Price': 'unit_price', 'Pizza_Count': 'quantity', 'Profit_Percentage': 'profit_percentage'}
df.rename(columns=names, inplace=True)

# Calculate additional columns
df['gross_margin'] = df['profit_percentage'] / 100
df['gross_profit'] = df['unit_price'] * (1 - df['gross_margin'])
df['cost'] = df['unit_price'] - df['gross_profit']

# Subset the dataframe for 'Individuale' product and create a copy
individuale_df = df[df['Product'] == 'Individuale'].copy()

# Check for and drop any rows with missing values
individuale_df = individuale_df.dropna(subset=['unit_price', 'quantity'])

# Extract x and y values
x_value = individuale_df['unit_price']
y_value = individuale_df['quantity']

# Ensure x and y are of the same length
assert len(x_value) == len(y_value), "x and y must have the same length"

# Add a constant to the independent variable (unit_price)
X = sm.add_constant(x_value)

# Fit the regression model
model = sm.OLS(y_value, X)
individuale_result = model.fit()

# Print the summary of the regression model
print(individuale_result.summary())

# Get the predicted values
predictions = individuale_result.predict(X)

# Add predictions to the dataframe
individuale_df['predictions'] = predictions

# Create a new range of unit prices within the observed data range
new_unit_prices = np.linspace(individuale_df['unit_price'].min(), individuale_df['unit_price'].max(), 100)
new_X = sm.add_constant(new_unit_prices)

# Predict the quantities for the new range of unit prices
new_predictions = individuale_result.predict(new_X)

# Calculate the gross profit for the new unit prices
# Assuming cost remains constant, using the mean cost from the original data
mean_cost = individuale_df['cost'].mean()
new_gross_profits = (new_unit_prices - mean_cost) * new_predictions

# Find the price that maximizes gross profit within the observed range
optimal_price_within_range = new_unit_prices[np.argmax(new_gross_profits)]
max_gross_profit_within_range = new_gross_profits.max()

# Print the optimal price and maximum gross profit within the observed range
print(f'Optimal Price (Within Observed Range): {optimal_price_within_range}')
print(f'Maximum Gross Profit (Within Observed Range): {max_gross_profit_within_range}')

# Plot the gross profit curve within the observed range
plt.figure(figsize=(10, 6))
plt.plot(new_unit_prices, new_gross_profits, label='Gross Profit Curve')
plt.scatter(optimal_price_within_range, max_gross_profit_within_range, color='red', label=f'Optimal Price: {optimal_price_within_range:.2f}')
plt.title('Gross Profit Curve (Within Observed Range)')
plt.xlabel('Unit Price')
plt.ylabel('Gross Profit')
plt.legend()
plt.grid(True)
plt.show()

# Plot the original gross profit curve
individuale_df['gross_profit_predictions'] = (individuale_df['unit_price'] - individuale_df['cost']) * individuale_df['predictions']
optimal_price_original = individuale_df.loc[individuale_df['gross_profit_predictions'].idxmax(), 'unit_price']
max_gross_profit_original = individuale_df['gross_profit_predictions'].max()

plt.figure(figsize=(10, 6))
plt.plot(individuale_df['unit_price'], individuale_df['gross_profit_predictions'], label='Gross Profit Curve')
plt.scatter(optimal_price_original, max_gross_profit_original, color='red', label=f'Optimal Price: {optimal_price_original:.2f}')
plt.title('Gross Profit Curve (Original Data)')
plt.xlabel('Unit Price')
plt.ylabel('Gross Profit')
plt.legend()
plt.grid(True)
plt.show()

# Compare the optimal prices and gross profits
print(f'Optimal Price (Original): {optimal_price_original}')
print(f'Maximum Gross Profit (Original): {max_gross_profit_original}')
