# Walmart Weekly Sales Prediction

## Project Overview

This machine learning project predicts Walmart weekly sales using store information, economic indicators, holiday information, and calendar features.

The project follows a complete data science lifecycle, including data collection, data cleaning, exploratory data analysis, feature engineering, preprocessing, model training, model comparison, model selection, serialization, and Streamlit deployment.

## Business Problem

Walmart stores need reliable weekly-sales estimates to support inventory planning, staffing, budgeting, and operational decision-making.

The objective of this project is to build a regression model that predicts weekly sales using historical store and economic data.

## Machine Learning Problem

- Problem type: Regression
- Target variable: `weekly_sales`
- Final model: Linear Regression

## Model Performance

The final Linear Regression model achieved approximately:

- MAE: £69,989.55
- RMSE: £101,451.25
- R² score: 0.9638

The model explained approximately 96.38% of the variation in weekly sales in the chronological test dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook
- Git and GitHub

## Project Structure

```text
walmart-sales-regression/
├── data/
├── models/
├── notebooks/
├── reports/images/
├── streamlit/
├── tests/
├── README.md
├── LICENSE
├── requirements.txt
└── startup.sh