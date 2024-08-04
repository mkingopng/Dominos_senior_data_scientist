import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('./../data/transformed_price_elasticity_data.csv')


# Define a function to calculate the quantity sold based on price using the linear model
def quantity_sold(price, model_params):
    return model_params['const'] + model_params['Menu_Price'] * price

# Define a function to calculate total gross profit
def total_gross_profit(price, model_params, cost_per_unit):
    quantity = quantity_sold(price, model_params)
    gross_profit_per_unit = price - cost_per_unit
    return quantity * gross_profit_per_unit

# Define a function to find the optimal price
def find_optimal_price(model_params, cost_per_unit):
    result = minimize(lambda price: -total_gross_profit(price, model_params, cost_per_unit), x0=5.0, bounds=[(0, 20)])
    return result.x[0]

# Load the model parameters
individuale_model_params = {'const': 6.6799, 'Menu_Price': -0.1705}
communita_model_params = {'const': 3.8124, 'Menu_Price': -0.0387}

# Define cost per unit for each product
individuale_cost_per_unit = data[data['Product'] == 'Individuale']['cost_per_unit'].mean()
communita_cost_per_unit = data[data['Product'] == 'Communita']['cost_per_unit'].mean()

# Find the optimal price for each product
optimal_price_individuale = find_optimal_price(individuale_model_params, individuale_cost_per_unit)
optimal_price_communita = find_optimal_price(communita_model_params, communita_cost_per_unit)

print(f"Optimal price for Individuale: ${optimal_price_individuale:.2f}")
print(f"Optimal price for Communita: ${optimal_price_communita:.2f}")

# Plotting the total gross profit function
prices = np.linspace(0, 20, 200)
profits_individuale = [total_gross_profit(price, individuale_model_params, individuale_cost_per_unit) for price in prices]
profits_communita = [total_gross_profit(price, communita_model_params, communita_cost_per_unit) for price in prices]

plt.figure(figsize=(12, 6))

plt.plot(prices, profits_individuale, label='Individuale', color='blue')
plt.axvline(optimal_price_individuale, color='blue', linestyle='--', label=f'Optimal Price Individuale (${optimal_price_individuale:.2f})')
plt.plot(prices, profits_communita, label='Communita', color='green')
plt.axvline(optimal_price_communita, color='green', linestyle='--', label=f'Optimal Price Communita (${optimal_price_communita:.2f})')

plt.xlabel('Price')
plt.ylabel('Total Gross Profit')
plt.title('Total Gross Profit as a Function of Price')
plt.legend()
plt.grid(True)
plt.show()
