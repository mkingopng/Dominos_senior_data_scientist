"""

"""
# linear_model_2.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats

# Load the data
data = pd.read_csv('./../data/transformed_price_elasticity_data.csv')

# Display the first few rows of the data
print(data.head())

# Split the data into Individuale and Communita products
individuale_data = data[data['Product'] == 'Individuale']
communita_data = data[data['Product'] == 'Communita']

# Function to build and evaluate a linear regression model using statsmodels
def build_linear_model(data, target_variable, feature_variable):
    # Add a constant to the feature variable
    X = sm.add_constant(data[feature_variable])
    y = data[target_variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the linear regression model
    model = sm.OLS(y_train, X_train).fit()

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate the model
    train_rmse = np.sqrt(((y_train - y_train_pred) ** 2).mean())
    test_rmse = np.sqrt(((y_test - y_test_pred) ** 2).mean())
    train_r2 = model.rsquared
    test_r2 = 1 - ((y_test - y_test_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()

    print(model.summary())
    print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    print(f"Train R^2: {train_r2}, Test R^2: {test_r2}")

    return model, X, y

# Function to calculate price elasticity of demand
def calculate_price_elasticity(model, data, feature_variable):
    # Calculate the coefficient
    price_elasticity = model.params[feature_variable]
    avg_price = data[feature_variable].mean()
    avg_sales = data['Pizza_Count'].mean()
    elasticity = price_elasticity * (avg_price / avg_sales)
    return elasticity

# Build and evaluate the model for Individuale using original data
print("Individuale Model (Original Data):")
individuale_model, X_individuale, y_individuale = build_linear_model(individuale_data, 'Pizza_Count', 'Menu_Price')

# Build and evaluate the model for Communita using original data
print("Communita Model (Original Data):")
communita_model, X_communita, y_communita = build_linear_model(communita_data, 'Pizza_Count', 'Menu_Price')

# Calculate price elasticity for Individuale (original data)
individuale_elasticity = calculate_price_elasticity(individuale_model, individuale_data, 'Menu_Price')
print(f"Individuale Price Elasticity (Original Data): {individuale_elasticity}")

# Calculate price elasticity for Communita (original data)
communita_elasticity = calculate_price_elasticity(communita_model, communita_data, 'Menu_Price')
print(f"Communita Price Elasticity (Original Data): {communita_elasticity}")

# Build and evaluate the model for Individuale using Box-Cox transformed data
print("Individuale Model (Box-Cox Transformed Data):")
individuale_model_bc, X_individuale_bc, y_individuale_bc = build_linear_model(individuale_data, 'Pizza_Count', 'BoxCox_Menu_Price')

# Build and evaluate the model for Communita using Box-Cox transformed data
print("Communita Model (Box-Cox Transformed Data):")
communita_model_bc, X_communita_bc, y_communita_bc = build_linear_model(communita_data, 'Pizza_Count', 'BoxCox_Menu_Price')

# Calculate price elasticity for Individuale (Box-Cox transformed data)
individuale_elasticity_bc = calculate_price_elasticity(individuale_model_bc, individuale_data, 'BoxCox_Menu_Price')
print(f"Individuale Price Elasticity (Box-Cox Transformed Data): {individuale_elasticity_bc}")

# Calculate price elasticity for Communita (Box-Cox transformed data)
communita_elasticity_bc = calculate_price_elasticity(communita_model_bc, communita_data, 'BoxCox_Menu_Price')
print(f"Communita Price Elasticity (Box-Cox Transformed Data): {communita_elasticity_bc}")

# Plotting the results
plt.figure(figsize=(14, 12))

# Scatter plot and fitted line for Individuale (original data)
plt.subplot(2, 2, 1)
plt.scatter(individuale_data['Menu_Price'], individuale_data['Pizza_Count'], color='blue', label='Individuale - Actual')
plt.plot(individuale_data['Menu_Price'], individuale_model.predict(sm.add_constant(individuale_data['Menu_Price'])), color='red', linewidth=2, label='Individuale - Fitted')
plt.scatter(communita_data['Menu_Price'], communita_data['Pizza_Count'], color='green', label='Communita - Actual')
plt.plot(communita_data['Menu_Price'], communita_model.predict(sm.add_constant(communita_data['Menu_Price'])), color='orange', linewidth=2, label='Communita - Fitted')
plt.xlabel('Menu_Price')
plt.ylabel('Pizza_Count')
plt.title('Menu_Price vs Pizza_Count (Original Data)')
plt.legend()

# Residual plots (original data)
plt.subplot(2, 2, 2)
residuals_individuale = y_individuale - individuale_model.predict(X_individuale)
residuals_communita = y_communita - communita_model.predict(X_communita)
plt.scatter(individuale_model.predict(X_individuale), residuals_individuale, color='blue', label='Individuale Residuals')
plt.scatter(communita_model.predict(X_communita), residuals_communita, color='green', label='Communita Residuals')
plt.axhline(y=0, color='red', linewidth=2)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot (Original Data)')
plt.legend()

# Scatter plot and fitted line for Individuale (Box-Cox transformed data)
plt.subplot(2, 2, 3)
plt.scatter(individuale_data['BoxCox_Menu_Price'], individuale_data['Pizza_Count'], color='blue', label='Individuale - Actual')
plt.plot(individuale_data['BoxCox_Menu_Price'], individuale_model_bc.predict(sm.add_constant(individuale_data['BoxCox_Menu_Price'])), color='red', linewidth=2, label='Individuale - Fitted')
plt.scatter(communita_data['BoxCox_Menu_Price'], communita_data['Pizza_Count'], color='green', label='Communita - Actual')
plt.plot(communita_data['BoxCox_Menu_Price'], communita_model_bc.predict(sm.add_constant(communita_data['BoxCox_Menu_Price'])), color='orange', linewidth=2, label='Communita - Fitted')
plt.xlabel('BoxCox_Menu_Price')
plt.ylabel('Pizza_Count')
plt.title('BoxCox_Menu_Price vs Pizza_Count (Transformed Data)')
plt.legend()

# Residual plots (Box-Cox transformed data)
plt.subplot(2, 2, 4)
residuals_individuale_bc = y_individuale_bc - individuale_model_bc.predict(X_individuale_bc)
residuals_communita_bc = y_communita_bc - communita_model_bc.predict(X_communita_bc)
plt.scatter(individuale_model_bc.predict(X_individuale_bc), residuals_individuale_bc, color='blue', label='Individuale Residuals')
plt.scatter(communita_model_bc.predict(X_communita_bc), residuals_communita_bc, color='green', label='Communita Residuals')
plt.axhline(y=0, color='red', linewidth=2)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot (Box-Cox Transformed Data)')
plt.legend()

plt.tight_layout()
plt.show()

# Save the models if needed
import joblib
joblib.dump(individuale_model, 'individuale_model.pkl')
joblib.dump(communita_model, 'communita_model.pkl')
joblib.dump(individuale_model_bc, 'individuale_model_bc.pkl')
joblib.dump(communita_model_bc, 'communita_model_bc.pkl')
