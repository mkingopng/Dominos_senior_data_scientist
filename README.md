# Dominos_senior_data_scientist

1. Understand the Dataset

    Dataset Columns:
        Product: Type of pizza (Individuale or Communita).
        Menu_Price: Price of the pizza.
        Pizza_Count: Number of pizzas sold.
        Profit_Percentage: Profit percentage from the sales.

2. Data Exploration and Cleaning

    **Load and Inspect Data**:
    - Load the dataset using pandas and inspect the first few rows to understand its structure.
    - Check for missing values and handle them appropriately.
    - Ensure data types are correct for analysis.

3. Exploratory Data Analysis (EDA)

    **Summary Statistics**:
        Calculate summary statistics (mean, median, standard deviation) for each variable.
    **Visualizations**:
        Plot the distribution of prices, sales, and profit percentages for both products.
        Use scatter plots to visualize the relationship between price and sales.
        Use line plots to show trends over different price points.

4. Statistical Analysis

    **Correlation Analysis**: Calculate the correlation between price and sales for both products.
    **Price Elasticity Calculation**:
    - Calculate the price elasticity of demand using the formula:
    - Price Elasticity = % change in quantity demanded / % change in price
    - Price Elasticity= % change in price% change in quantity demanded
    - Determine which product is more price elastic.

5. Machine Learning Approach

    **Feature Engineering**: Create features such as the square of the price, interaction terms, etc.
    Model Building:
    - Split the data into training and testing sets.
    - Train linear regression models to predict sales based on price.
    - Train more complex models (e.g., polynomial regression, decision trees,
	  or random forests or GBDT) to capture non-linear relationships.
    
    **Model Evaluation**:
    - Evaluate models using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.
    - Compare models to choose the best one for predicting sales.

6. Optimal Pricing Strategy
    **Optimization**:
        - Use the models to predict sales at different price points.
        - Calculate the profit for each price point and identify the price that maximizes profit.
    **Validation**: Validate the optimal price points with a subset of data or using cross-validation techniques.

7. Generate Insights
    **Comparison**: Compare the price elasticity and optimal prices for both products.
    **Additional Insights**: Identify any trends or patterns in the data that could inform other business strategies (e.g., seasonal trends, peak sales periods).
    **Recommendations**: Provide actionable recommendations based on the analysis (e.g., pricing strategies, promotional offers).

8. Presentation Preparation

    PowerPoint Deck:
        Slide 1: Title slide with the purpose of the analysis.
        Slide 2: Summary of the dataset and key variables.
        Slide 3: Visualizations showing the relationship between price and sales.
        Slide 4: Statistical analysis results (correlation, price elasticity).
        Slide 5: Machine learning model results and optimal pricing strategy.
        Slide 6: Additional insights and recommendations.
        Slide 7: Conclusion and next steps.
    Narrative:
        Prepare a narrative to explain each slide in a clear and concise manner.
        Anticipate questions from the board and prepare answers.

9. Code Implementation (Python)

    Libraries to Use:
        pandas for data manipulation.
        numpy for numerical operations.
        matplotlib and seaborn for visualizations.
        scipy for statistical analysis.
        sklearn for machine learning models.
