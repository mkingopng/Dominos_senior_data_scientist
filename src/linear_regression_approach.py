"""
linear regression modelling for case study
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from tqdm import tqdm
import logging
from typing import Callable
from functools import wraps
import time
import datetime

today = datetime.datetime.now()

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
file_handler = logging.FileHandler(f'./../logs/{today}product_analysis.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
)
logging.getLogger().addHandler(file_handler)

# Set global plotting parameters using rcParams
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.loc'] = 'upper left'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 6
plt.rcParams['font.size'] = 12


def timing(f: Callable) -> Callable:
    """
    Decorator to log the time taken by a function
    :param f:
    :return:
    """
    @wraps(f)
    def wrap(*args, **kw) -> Callable:
        """
        Wrapper function to log the time taken by a function
        :param args:
        :param kw:
        :return:
        """
        start_time = time.time()
        result = f(*args, **kw)
        end_time = time.time()
        logging.info(f'Function {f.__name__} took {end_time-start_time:.3f} seconds')
        return result
    return wrap


def log_function_call(f: Callable) -> Callable:
    """
    Decorator to log the function calls
    :param f:
    :return:
    """
    @wraps(f)
    def wrap(*args, **kw) -> Callable:
        """
        Wrapper function to log the function calls
        :param args:
        :param kw:
        :return:
        """
        logging.info(f'Function {f.__name__} called')
        return f(*args, **kw)
    return wrap


class ProductAnalysis:
    def __init__(self, data: pd.DataFrame, product_name: str) -> None:
        self.data = data[data['Product'] == product_name].copy()
        self.product_name = product_name
        self.model = None
        self.coefficients = None

    @log_function_call
    @timing
    def preprocess_data(self) -> None:
        pass  # assuming data is preprocessed before being loaded

    @log_function_call
    @timing
    def plot_pair_plot(self) -> None:
        sns.pairplot(self.data)
        plt.show()

    @log_function_call
    @timing
    def plot_histograms(self) -> None:
        fig, axes = plt.subplots(2, 3)
        fig.suptitle(f'{self.product_name} Histograms')

        sns.histplot(self.data['BoxCox_Menu_Price'], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('BoxCox Menu Price')

        sns.histplot(self.data['Pizza_Count'], kde=True, ax=axes[0, 1])
        axes[0, 1].set_title('BoxCox Pizza Count')

        sns.histplot(self.data['Profit_Percentage'], kde=True, ax=axes[0, 2])
        axes[0, 2].set_title('Profit Percentage')

        # Hide unused subplots
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)
        axes[1, 2].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

    @log_function_call
    @timing
    def calculate_correlation(self) -> None:
        correlation = self.data.drop(columns=['Product']).corr()
        logging.info(
            f"\nCorrelation matrix for {self.product_name}:\n{correlation}")

    @log_function_call
    @timing
    def calculate_price_elasticity(self) -> None:
        self.data = self.data.sort_values('BoxCox_Menu_Price')
        self.data['Price_Change'] = self.data['BoxCox_Menu_Price'].pct_change()
        self.data['Count_Change'] = self.data[
            'Pizza_Count'].pct_change()
        self.data['Elasticity'] = self.data['Count_Change'] / self.data[
            'Price_Change']
        # todo: print the price elasticity of demand function using pprint

    @log_function_call
    @timing
    def train_model(self) -> None:
        self.data['BoxCox_Price_Squared'] = self.data['BoxCox_Menu_Price'] ** 2

        X = self.data[['BoxCox_Menu_Price', 'BoxCox_Price_Squared']]
        y = self.data['Pizza_Count']

        # Split data into training, testing, and evaluation sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=0.4,
            random_state=42
        )
        X_test, X_eval, y_test, y_eval = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=42
        )

        # Initialize the model
        self.model = LinearRegression()

        # Train the model on the training set
        self.model.fit(X_train, y_train)

        # Evaluate on the test set
        y_pred_test = self.model.predict(X_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
        r2_test = r2_score(y_test, y_pred_test)
        logging.info(f'\nModel Performance on Test Set for {self.product_name}:\nMAE: {mae_test}\nMSE: {mse_test}\nRMSE: {rmse_test}\nR-squared: {r2_test}')

        # Perform cross-validation on the test set
        scoring = {
            'MAE': make_scorer(mean_absolute_error),
            'MSE': make_scorer(mean_squared_error),
            'RMSE': make_scorer(
                lambda y_true, y_pred: mean_squared_error(
                    y_true,
                    y_pred,
                    squared=False
                )
            ),
            'R2': make_scorer(r2_score)
        }
        cv_results = cross_validate(
            self.model,
            X_test,
            y_test,
            cv=5,
            scoring=scoring,
            return_train_score=False
        )
        logging.info(f'\nCross-Validation Results for {self.product_name}:')
        for metric in scoring.keys():
            scores = cv_results[f'test_{metric}']
            logging.info(
                f'{metric} Scores: {scores}\nMean {metric}: {scores.mean()}')

        # Evaluate the model on the evaluation set
        y_pred_eval = self.model.predict(X_eval)
        mae_eval = mean_absolute_error(y_eval, y_pred_eval)
        mse_eval = mean_squared_error(y_eval, y_pred_eval)
        rmse_eval = mean_squared_error(y_eval, y_pred_eval, squared=False)
        r2_eval = r2_score(y_eval, y_pred_eval)
        logging.info(
            f'\nEvaluation on Evaluation Set for {self.product_name}:\nMAE: {mae_eval}\nMSE: {mse_eval}\nRMSE: {rmse_eval}\nR-squared: {r2_eval}')

    @log_function_call
    @timing
    def plot_optimal_price(self) -> None:
        """
        Plot the optimal price point for the product.
        :return:
        """
        prices = np.linspace(self.data['Menu_Price'].min(),
                             self.data['Menu_Price'].max(), 100)
        boxcox_prices = (
                    prices ** 0.5)  # Assuming Box-Cox transformation (adjust this based on actual transformation used)
        boxcox_prices_squared = boxcox_prices ** 2

        prices_df = pd.DataFrame({
            'Menu_Price': prices,
            'BoxCox_Menu_Price': boxcox_prices,
            'BoxCox_Price_Squared': boxcox_prices_squared
        })

        sales = self.model.predict(
            prices_df[['BoxCox_Menu_Price', 'BoxCox_Price_Squared']])

        profit = (prices - self.data['cost_per_unit'].mean()) * sales
        optimal_index = np.argmax(profit)
        optimal_price = prices[optimal_index]

        self.coefficients = np.polyfit(prices, sales, 2)
        poly_func = np.poly1d(self.coefficients)
        first_derivative_func = np.polyder(poly_func)
        first_derivative_at_optimal_price = first_derivative_func(
            optimal_price)

        fig, ax1 = plt.subplots()

        ax1.plot(prices, sales, label='Quantity Sold')
        ax1.axvline(x=optimal_price, color='red', linestyle='--',
                    label=f'Optimal Price: {optimal_price:.2f}')
        ax1.set_xlabel('Price')
        ax1.set_ylabel('Quantity Sold')
        ax1.set_title(f'{self.product_name} Optimal Price Point')

        ax2 = ax1.twinx()
        ax2.plot(prices, first_derivative_func(prices),
                 label='First Derivative', color='green')
        ax2.axhline(y=0, color='gray', linestyle='--', label='Zero Slope')
        ax2.set_ylabel('First Derivative')

        fig.legend(loc='upper left')
        plt.xlim(self.data['Menu_Price'].min(),
                 self.data['Menu_Price'].max() * 1.5)  # Extend x-axis
        plt.show()


if __name__ == '__main__':
    # Load Data
    data = pd.read_csv('./../data/transformed_price_elasticity_data.csv')

    # Wrap in tqdm to show progress bars
    with tqdm(total=2, desc="Processing Data") as pbar:
        # Separate the data by Product and analyze each product separately
        pbar.set_description("Processing Individuale")
        individuale_analysis = ProductAnalysis(data, 'Individuale')
        individuale_analysis.preprocess_data()
        individuale_analysis.calculate_correlation()
        individuale_analysis.calculate_price_elasticity()
        individuale_analysis.train_model()
        pbar.update()

        pbar.set_description("Processing Communita")
        communita_analysis = ProductAnalysis(data, 'Communita')
        communita_analysis.preprocess_data()
        communita_analysis.calculate_correlation()
        communita_analysis.calculate_price_elasticity()
        communita_analysis.train_model()
        pbar.update()

    # Plotting
    with tqdm(total=2, desc="Plotting Data") as pbar:
        pbar.set_description("Plotting Individuale")
        individuale_analysis.plot_pair_plot()
        individuale_analysis.plot_histograms()
        individuale_analysis.plot_optimal_price()
        pbar.update()

        pbar.set_description("Plotting Communita")
        communita_analysis.plot_pair_plot()
        communita_analysis.plot_histograms()
        communita_analysis.plot_optimal_price()
        pbar.update()


# todo: EDA,
#  feature engineering,
#  logging,
#  plot price and quantity to get PED
#  fix plots: extend curve, note optimal price
#  save plots,
#  hyperparameters,
#  should include price quantity and profit,
#  docstrings,
#  type hints,
#  decorators,
#  unit tests
