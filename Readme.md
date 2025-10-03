
    # Prediction and Dashboard for Pharma Sales and Price.
    **Objective**: This Project focuses on the prediction of Pharma Sales and Price as well as a detailed Dashboard Analysis.

    ## Table of Contents
    - [Data Overview](#data-overview)
    - [Data Cleaning](#data-cleaning)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Machine Learning Models](#machine-learning-models)
    - [Data Preprocessing](#data-preprocessing)
    - [Feature Engineering](#feature-engineering)
    - [Model Training & Evaluation](#model-training--evaluation)
    - [Feature Importance](#feature-importance)
    - [Predictions on New Data](#predictions-on-new-data)
    - [Model Artifacts](#model-artifacts)
    - [Dashboard Creation](#dashboard-creation)
    - [Conclusion](#conclusion)

    ## Data Overview
        The dataset contains sales records from a pharmaceutical company, including details about products, distributors, sales representatives, sales and price figures.
    The original dataset has 254,083 rows and 18 columns. After cleaning, it has 218,928 and 18 columns. The cleaned dataset is further used for analysis, modeling, and 
    also used in the creation of a dashboard report PowerBI.
    The cleaned dataset is saved as 'cleaned_pharma_data.csv', and the original dataset is 'pharma-data.csv'. 
    **Columns**:
    - Distributor
    - Customer Name
    - City
    - Country
    - Channel
    - Sub-channel
    - Product Name
    - Product Class
    - Quantity
    - Price
    - Sales
    - Month
    - Year
    - Name of Sales Rep
    - Manager
    - Sales Team
    - Quantity_Price (Engineered Feature)

    ## Data Cleaning
    The data cleaning process involved:
    - Handling errors in 'Quantity' and 'Sales' columns by converting to absolute values.
    - Fixing data type error by converting 'Quantity', 'Sales', and 'Price' columns to numeric types.
    - Removing duplicates (there were 4 duplicates).
    - Identifying and removing outliers in 'Sales' and 'Quantity' columns using the IQR method.
    - Final cleaned dataset has 218,928 rows and 18 columns.

    ## Exploratory Data Analysis (EDA)
    EDA was performed to understand the data distribution and relationships:
    - Descriptive statistics were generated for numerical columns.
    - Correlation analysis showed strong correlation between 'Sales' and 'Price'. However, 'Quantity' had the strongest correlation with 'Sales'.
    - Visualizations included bar plots, pie charts, and line plots to analyze sales by product name and class, channel, year, month, distributor, and sales rep and team.
    - Further analysis was done to understand the quantity sold by distributors.
    - Key insights were drawn from the visualizations to inform feature selection and engineering.
    Visuals created during EDA were saved as PNG files and are saved in the "visual outputs" directory.

    ## Machine Learning Models
    Two separate XGBoost regression models were developed:
    1. **Price Prediction Model**: Predicts the price of pharmaceutical products based on various features but excluding 'Quanity', 'Quantity_Price', and 'Sales'.
    2. **Sales Prediction Model**: Predicts sales figures using all relevant features including 'Quantity', 'Price' and 'Quantity_Price'.
    Both models were trained and evaluated using metrics such as RMSE, MAE, and R2 Score.

    ## Data Preprocessing
    Data preprocessing steps included:
    - Encoding categorical variables using Label Encoding.
    - Scaling numerical features ('Quantity' and 'Year') using StandardScaler. Taget variables ('Price' and 'Sales') were not scaled.
    - Splitting the data into training and testing sets (80-20 split). The random state was set to 42 for better results.

    ## Feature Engineering
    Although by default 'Sales' is as a result of multiplying 'Price' by 'Quantity', an additional feature 'Quantity_Price' was created by multiplying 'Quantity' and 'Price'.
    This feature was used in the Sales Prediction Model to improve the accuracy of the sales model and predictions.

    ## Model Training & Evaluation
    Both models were trained using the XGBoost regressor with hyperparameters optimized through cross-validation.
    The models were evaluated on the test set using:
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - R2 Score
    - Cross-validation results were also analyzed to ensure model robustness and inspect for overfitting or underfitting.
    The Sales model suffered from overfitting while the price model had slight underfitting. Hyperparameters were adjusted to mitigate these issues.
    After the hyperparameter tuning, the models were able to generalize well on unseen data and the issues of overfitting and underfitting were resolved.
    However, the sales model still has errors (MSE, MAE) due to the complexity of sales data.
    The evaluations metrics for both models are summarized below:
    - **Price Model**:
    - mse_price = 0.00
    - rmse_price = 0.00
    - mae_sales = 0.00
    - r2_sales = 1.00
    - Cross-validation for price model:
    cv_price =     train-rmse-mean  train-rmse-std  test-rmse-mean  test-rmse-std
95         0.000352        0.000099        0.000352       0.000100
96         0.000320        0.000099        0.000320       0.000100
97         0.000285        0.000086        0.000286       0.000087
98         0.000260        0.000078        0.000261       0.000078
99         0.000239        0.000076        0.000239       0.000077
    - **Sales Model**:
    - mse_sales = 9486.45   
    - rmse_sales = 97.40 
    - mae_sales = 46.58
    - r2_sales = 1.00
    - Cross-validation for sales model:
    cv_sales =     train-rmse-mean  train-rmse-std  test-rmse-mean  test-rmse-std
95        99.073010        0.598024      101.209234       1.627515
96        98.868545        0.598998      101.036722       1.588280
97        98.770108        0.601522      100.946309       1.546351
98        98.549451        0.633928      100.761270       1.536726
99        98.236630        0.645569      100.526271       1.509237

    ## Feature Importance
    Feature importance was analyzed for both models to understand which features had the most impact on predictions.
    - For the Price Model, 'Product Name' and 'Product Class' were the most important features.
    - For the Sales Model, 'Price', 'Quantity', 'Product Class' and 'Product Name' were the top features.
    Visualizations of feature importance were created to aid interpretation.

    ## Predictions on New Data
    The trained models were used to make predictions on existing data points and also on new data points. 
    Due to the complexity of sales data, the sales model had some errors in predictions so the price model was used to predict price and then the results for sales was 
    calculated by multiplying the predicted price by the quantity.
    An example of existing data used for predictions:
    - Distributor: 'Gottlieb-Cruickshank'
    - Customer Name: 'Keeling LLC Pharmacy'
    - City: 'Olsztyn'
    - Country: 'Poland'
    - Channel: 'Pharmacy'
    - Sub-channel: 'Private'
    - Product Name: 'Oxymotroban Fexoformin'
    - Product Class: 'Analgesics'
    - Quantity: 20.0
    - Month: 'January'
    - Year: 2018
    - Name of Sales Rep: 'Anne Wu'
    - Manager: 'Britanny Bold'
    - Sales Team: 'Delta'
        The predicted results were:
    - Predicted Sales ($): 9160
    - Predicted Price ($): 458
    The results were rounded to the nearesr integer for simplicity and they were exactly the same as the actual values.

    ## Model Artifacts
    All components of the pipeline were saved using joblib for future use. This includes:
    - Trained models (Sales and Price)
    - Label encoders for categorical variables
    - StandardScaler for numerical features
    - Feature lists for both models
    - Feature importance dataframes for both models.
    The artifacts are saved in the 'model_artifacts' directory.

    ## Dashboard Creation
    A dashboard was created using PowerBI to visualize key insights from the data. The dashboard includes:
    - Sales by Product Class
    - Sales by Channel and Sub-channel
    - Sales by Year and Month
    - Sales by Distributor and Country
    - Sales by Sales Rep and Sales Team
    The dashboard provides an interactive way to explore the data and understand sales trends.
    The cleaned dataset 'cleaned_pharma_data.csv' was used as the data source for the dashboard.
    The PowerBI dashboard file is saved as 'Pharma Sales Dashboard.pbix'.

    ## Conclusion
    This project successfully demonstrated the end-to-end process of data cleaning, exploratory data analysis, feature engineering, model training and evaluation, 
    and dashboard creation for pharmaceutical sales data. 
    The developed models can be used for predicting sales and prices, and the dashboard provides valuable insights for decision-making.

    **Note**: The models, while effective, may still have limitations due to the inherent complexity and variability in sales data.

    **Connect with me**:
    - [LinkedIn](https://www.linkedin.com/in/ireoluwawolemi-akindipe-16b711373?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
    - [GitHub](https://github.com/Dhela456)

