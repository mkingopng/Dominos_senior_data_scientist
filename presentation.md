---
marp: true
theme: Night
paginate: true
backgroundSize: contain
---

# Title: Analysis of Product Pricing and Sales
- **Subtitle**: Insights and Recommendations
- **Presenter**: Michael Kingston
- **Date**:

---
# Relationship Between Product Price and Sales



## Communita Product:

![Communita Price vs. Sales Scatter Plot]()

## Individuale Product:

![Individuale Price vs. Sales Scatter Plot]()

<!--
In general terms we can express price elasticity of demand algebraically as:
$$\text{PED} = \dfrac{\Delta{Q}/Q}{\Delta{P}/P}$$
Where:
- $\Delta{Q}$ is the change in quantity demanded.
- $Q$ is the initial quantity demanded.
- $\Delta{P}$ is the change in price.
- $P$ is the initial price.

In general, the slope of the regression line is negative, indicating that as 
price increases, quantity demanded decreases. There are special exceptions to 
this, such as Giffen goods, but in our case we expect PED to be quite elastic
-->

---
<!-- Slide 3: Price Elasticity Comparison -->
# Price Elasticity of Demand

## Communita Product:
- Price elasticity: value
- Interpretation: 

## Individuale Product:
- Price elasticity: value
- Interpretation:

## Comparison: 
- Which product is more price elastic?: **answer**

**Visuals**: Tables or bullet points with elasticity values and interpretations.

---
<!-- Slide 4: Suitable Price Points -->
# Finding the Optimal Price

## Step 1: Calculate the Gross Profit Curve

The gross profit, \( G \), is given by:
\[ G = (P - C) \times Q \]
where:
- \( P \) is the unit price,
- \( C \) is the unit cost,
- \( Q \) is the quantity sold.

## Step 2: Calculate the Second Derivative and Set Its Slope to Zero

## Communita Plot 1:
![Communita Gross Profit Curve]()

## Individuale Plot 1:
![Individuale Optimal Price Point]('./../plots/individuale_gp_curve.png') 

!<--
To find the optimal price point, we need to take the second derivative of the 
gross profit function and set it to zero.

1. **First Derivative**: Calculate the first derivative of the gross profit 
   function with respect to the price:
   \[ G'(P) = \frac{d}{dP} \left( (P - C) \times Q \right) \]

2. **Second Derivative**: Calculate the second derivative of the gross profit 
   function \[ G''(P) = \frac{d^2}{dP^2} \left( (P - C) \times Q \right) \]

3. **Set the Second Derivative to Zero**: Solve for \( P \) when the second 
   derivative is equal to zero to find the optimal price:
   \[ G''(P) = 0 \]

This means that the second derivative, which is tangential to the gross 
profit curve, has a slope of zero, it indicates the maximum point of the curve.

By solving this equation, we can determine the price $P$ that maximises the 
gross profit.

You'll note something interesting about these plots. The optimal price 
point based on this is the maximum value of Menu_Price. 

Further, the second derivative is not zero. 

Both points indicate that we have not achieve the optimal price point, as 
our dataset is limited to the range of prices in the dataset.
-->

---
## Communita Plot 2:
![Communita Optimal Price Point (extended)](./../plots/communita_ext_gp_curve.png)

## Individuale Plot 2:
![Individuale Optimal Price Point (Extended)]('./../plots/individuale_ext_gp_curve')

!<--
If we extend our predictions based on our equation, we can see that it 
predicts that the optimal price point falls outside the range of prices 
captured in the dataset. 

We have to be really careful about making predictions outside the range, so 
at this point we should at most note this and strongly suggest proper A/B 
testing, which we'll discuss in the next section.
-->

---
# Additional Insights for Growth

- **Experiment**: the first opportunity is to test the hypothesis that the 
optimal price falls outside the sample range. This can be done by running an A/B test.

- **More Data**:This is a very small sample, of only 157 records. 
- It contains no datetime data, 
- nor does it contain data of other factors that may impact sales aside from price.
- This may partly account for the low R2 value of the regression models, 
  indicating that there is information in variables not contained in this set

- **Customer Segmentation**: How different segments respond to pricing.
- **Seasonal Trends**: Seasonal sales patterns and strategy adjustments.
- **Product Bundling**: Benefits of bundling products.
- **Promotion Analysis**: Impact of past promotions and future strategies.

Visuals: Relevant charts or bullet points summarizing key insights.

---
# Further Work

---
# GitHub Repository

---
Sam has asked you to present the insights to his board of non-technical stakeholders

Further Instructions
- Prepare a presentation of around 20 minutes
- Capture the insights / learnings in a power point deck (5 -7 slides)
- Use your tool of choice (e.g. Excel, R and/or python)

