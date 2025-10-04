import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os


# custom
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        background-color: #333333;
        color: #ffffff;
        border: 1px solid #555555;
    }
    .stSelectbox > div > div > select {
        background-color: #333333;
        color: #ffffff;
        border: 1px solid #555555;
    }
    .stRadio > div {
        color: #ffffff;
    }
    .stSlider > div > div > div {
        background-color: #555555;
    }
    .stButton > button {
        background-color: #ff4040;
        color: #ffffff;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #cc0000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set Page Configuration
st.set_page_config(page_title="Pharma_Sales_App", layout='wide', initial_sidebar_state='auto', menu_items={
    'About': 'This web application is designed for the prediction of Pharmaceutical Sales!'
    })

@st.cache_data
# Load Data
def load_data():
    return pd.read_csv('C:/Users/IreOluwa/Documents/projects/Pharma-Data/Data/cleaned_pharma_data.csv')
data = load_data()
df = pd.DataFrame(data)

# Load model artifacts
pipeline = joblib.load('model_artifacts/pipeline.joblib')
features_for_sales = pipeline['features_for_sales']
features_for_price = pipeline['features_for_price']
label_encoders = pipeline['label_encoders']
scaler = pipeline['scaler']
model_sales = pipeline['model_sales']
model_price = pipeline['model_price']
categorical_columns = pipeline['categorical_columns']
numeric_columns = pipeline['numeric_columns']

# City to country mapping
city_to_country = {}
for city in label_encoders['City'].classes_:
    countries = df[df['City'].str.strip() == city]['Country'].str.strip().unique()
    city_to_country[city] = sorted(countries)[0] if len(countries) > 0 else 'Unknown'
    if len(countries) > 1:
        st.warning(f"City '{city}' maps to multiple countries: {countries}. Using '{city_to_country[city]}'.")
# Product Name to Product Class mapping
product_to_class = {}
for product in label_encoders['Product Name'].classes_:
    classes = df[df['Product Name'].str.strip() == product]['Product Class'].str.strip().unique()
    product_to_class[product] = sorted(classes)[0] if len(classes) > 0 else 'Unknown'
    if len(classes) > 1:
        st.warning(f"Product '{product}' maps to multiple classes: {classes}. Using '{product_to_class[product]}'.")

# Prediction Functions
def predict_new_data(new_data, model_sales, model_price, label_encoders, scaler, features_for_sales,
                     features_for_price, categorical_columns, numeric_columns):
    new_data_df = pd.DataFrame(new_data)
    # Encode categorical columns
    for col in categorical_columns:
        if col in new_data_df.columns:
            le = label_encoders[col]
            # Add 'Unknown' to classes if needed
            if 'Unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'Unknown')
            new_data_df[col] = new_data_df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
            new_data_df[col] = le.transform(new_data_df[col])
    
    # Scale numeric features
    input_data_as_array = np.asarray(new_data_df[numeric_columns])
    std_data = scaler.inverse_transform(input_data_as_array.reshape(1, -1)).flatten()
    new_data_df[numeric_columns] = std_data

    # Select only the columns used for training, in the correct order
    X_sales_pred = new_data_df.reindex(columns=features_for_sales, fill_value=0)
    X_price_pred = new_data_df.reindex(columns=features_for_price, fill_value=0)
    # Predict sales and price
    price_prediction = model_price.predict(X_price_pred)
    sales_prediction = price_prediction * new_data_df['Quantity']

    return sales_prediction, price_prediction

# Streamlit App
st.markdown('<style>div.block-container{padding-top: 2rem;}</style>', unsafe_allow_html=True)
#st.title("Pharma Sales and Price Prediction :red[App]")
st.markdown("<div style='text-align: center; font-size: 50px; font-weight: bold;'>Pharma Sales and Price Prediction <span style='color: red; font-size: 50px; font-weight: bold;'>App</span></div>", unsafe_allow_html=True)
st.write("Enter the details below to predict Sales and Price.")

# create input form
with st.form(key='prediction_form'):
    st.subheader("Input Features")
    col1, col2, col3 = st.columns(3)
    with col1:
    # Categorical inputs
        distributor = st.selectbox("Distributor", options=label_encoders['Distributor'].classes_)
        customer_name = st.selectbox("Customer Name", options=label_encoders['Customer Name'].classes_)
        city = st.selectbox("City", options=label_encoders['City'].classes_)
        country = st.selectbox("Country", options=label_encoders['Country'].classes_, 
                           index=label_encoders['Country'].classes_.tolist().index(city_to_country[city])
                           if city_to_country[city] in label_encoders['Country'].classes_ else 0)
    with col2:
        channel = st.selectbox('Channel', options=label_encoders['Channel'].classes_)
        sub_channel = st.selectbox('Sub-channel', options=label_encoders['Sub-channel'].classes_)
        product_name = st.selectbox('Product Name', options=label_encoders['Product Name'].classes_)
        product_class = st.selectbox("Product Class", options=label_encoders['Product Class'].classes_,
                                 index=label_encoders['Product Class'].classes_.tolist().index(product_to_class[product_name])
                                 if product_to_class[product_name] in label_encoders['Product Class'].classes_ else 0)
    with col3:
        month = st.selectbox('Month', options=label_encoders['Month'].classes_)
        name_of_sales_rep = st.selectbox('Name of Sales Rep', options=label_encoders['Name of Sales Rep'].classes_)
        manager = st.selectbox('Manager', options=label_encoders['Manager'].classes_)
        sales_team = st.selectbox('Sales Team', options=label_encoders['Sales Team'].classes_)

    # Numeric inputs
    quantity = st.number_input('Quantity', min_value=0, max_value=117600)
    year = st.number_input('Year', min_value=2017, max_value=2020)
    #price = st.number_input('Price', min_value=0, max_value=794)

    # Optional actuals for comparison
    actual_sales = st.number_input('Actual Sales (optional)', value=0.0)
    actual_Price = st.number_input('Actual Price (optional)', value=0.0)
    submit_button = st.form_submit_button(label='Predict')

# Process form submission
if submit_button:
    new_data_df= {
        'Distributor': [distributor],
        'Customer Name': [customer_name],
        'City': [city],
        'Country': [country],
        'Channel': [channel],
        'Sub-channel': [sub_channel],
        'Product Name': [product_name],
        'Product Class': [product_class],
        'Quantity': [quantity],
        'Month': [month],
        'Year': [year],
        'Name of Sales Rep': [name_of_sales_rep],
        'Manager': [manager],
        'Sales Team': [sales_team]
       # 'Quantity_Price': quantity * price_prediction
#        'Price': [price]  
    }
    if actual_Price > 0:
        new_data_df['Sales'] = [actual_Price]
    if actual_sales > 0:
        new_data_df['Price'] = [actual_sales]

# Predictions
    sales_prediction, price_prediction = predict_new_data(new_data_df, model_sales, model_price, label_encoders, scaler, features_for_sales, features_for_price, categorical_columns, numeric_columns)
    results = pd.DataFrame({
        'Predicted Sales ($)': sales_prediction.round(2),
    #  'Actual Sales ($)': new_data_df['Sales'],
        'Predicted Price ($)': price_prediction.round(2),
    #  'Actual Price ($)': new_data_df['Price']
    })
    for col in results.columns:
        results[col] = results[col].astype(int)
    results = results.astype(str).replace('\.0', '', regex=True)
    st.subheader("Prediction Results")
    st.table(results)