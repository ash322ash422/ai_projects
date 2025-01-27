import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# Load the trained model
cwd = Path.cwd()
print(f"cwd:{cwd}")
# model_path = cwd / 'model.pkl'
model_path = cwd / 'tutorial' / 'tut_ml' / 'bigmart_sales_prediction' / 'model.pkl'
trained_model = joblib.load(model_path)

# Title for the app
st.title('Sales Prediction')


# Get numerical input features from the user
numerical_features = ['Item_Weight', 'Item_MRP']
input_features = []

for feature_name in numerical_features:
    value = st.number_input(f"Enter value for {feature_name}", step=0.01)
    input_features.append(value)

# Define mappings for categorical features
Item_Fat_Content_dict = {'Low Fat': 0, 'Regular': 1}
Item_Type_dict = {
    'Fruits and Vegetables': 0, 'Snack Foods': 1, 'Household': 2, 'Frozen Foods': 3,
    'Dairy': 4, 'Canned': 5, 'Baking Goods': 6, 'Health and Hygiene': 7,
    'Soft Drinks': 8, 'Meat': 9, 'Breads': 10, 'Hard Drinks': 11, 'Others': 12,
    'Starchy Foods': 13, 'Breakfast': 14, 'Seafood': 15
}
Outlet_Size_dict = {'Medium': 0, 'Small': 1, 'High': 2}
Outlet_Location_Type_dict = {'Tier 3': 0, 'Tier 2': 1, 'Tier 1': 2}
Outlet_Type_dict = {
    'Supermarket Type1': 0, 'Grocery Store': 1, 'Supermarket Type3': 2, 'Supermarket Type2': 3
}

# Dropdown menus for categorical features
feature_texts = [
    ('Item_Fat_Content', Item_Fat_Content_dict),
    ('Item_Type', Item_Type_dict),
    ('Outlet_Size', Outlet_Size_dict),
    ('Outlet_Location_Type', Outlet_Location_Type_dict),
    ('Outlet_Type', Outlet_Type_dict)
]

for feature_name, feature_dict in feature_texts:
    value = st.selectbox(f"Select value for {feature_name}:", list(feature_dict.keys()))
    input_features.append(feature_dict[value])

# Predict button
if st.button('Predict'):
    # Convert input features to a NumPy array
    input_features_array = np.array(input_features).reshape(1, -1)

    # Make predictions using the loaded model
    predicted_sales = trained_model.predict(input_features_array)

    st.write(f"Predicted sales: {predicted_sales[0]:.2f}")
