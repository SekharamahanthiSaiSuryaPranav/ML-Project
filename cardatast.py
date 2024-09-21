import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import streamlit as st

# Load your dataset
data = pd.read_csv('car details v4.csv')

# Feature Engineering: Age of the car
data['Car_Age'] = 2024 - data['Year']  # Assuming current year is 2024
data.drop('Year', axis=1, inplace=True)  # Drop Year column if not needed

# Prepare the features and target variable
X = pd.get_dummies(data.drop('Price', axis=1), drop_first=True)
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestRegressor()

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, r'C:\Users\srivi\Downloads\car file\car_price_predictor_model.pkl')

# Load the trained model for prediction
model = joblib.load(r'C:\Users\srivi\Downloads\car file\car_price_predictor_model.pkl')

# Load the columns of the original dataset (for alignment in one-hot encoding)
X_columns = pd.get_dummies(data.drop('Price', axis=1), drop_first=True).columns

# Define the prediction function
def predict_price(year, make, model_name, kilometer, engine, max_power, fuel_type, transmission, seating_capacity):
    user_data = {
        'Make': make,
        'Model': model_name,
        'Kilometer': kilometer,
        'Engine': engine,
        'Max Power': max_power,
        'Fuel Type': fuel_type,
        'Transmission': transmission,
        'Seating Capacity': seating_capacity,
        'Car_Age': 2024 - year
    }

    user_df = pd.DataFrame([user_data])
    user_df = pd.get_dummies(user_df)
    user_df = user_df.reindex(columns=X_columns, fill_value=0)

    predicted_price = model.predict(user_df)
    
    return predicted_price[0]

# Streamlit UI
st.title('Car Price Predictor')

# Collect user inputs
year = st.number_input('Year of Manufacture', min_value=1990, max_value=2024, step=1)
make = st.text_input('Car Make (e.g., Toyota, Honda)')
model_name = st.text_input('Car Model (e.g., Corolla, Civic)')
kilometer = st.number_input('Kilometers Driven', min_value=0, step=500)
engine = st.number_input('Engine Capacity (cc)', min_value=500, step=100)
max_power = st.number_input('Max Power (bhp)', min_value=30, step=10)
fuel_type = st.selectbox('Fuel Type', ('Petrol', 'Diesel'))
transmission = st.selectbox('Transmission Type', ('Manual', 'Automatic'))
seating_capacity = st.number_input('Seating Capacity', min_value=2, max_value=10, step=1)

# Predict button
if st.button('Predict'):
    predicted_price = predict_price(year, make, model_name, kilometer, engine, max_power, fuel_type, transmission, seating_capacity)
    st.success(f'The predicted price of the car is: ₹{predicted_price:,.2f}')
