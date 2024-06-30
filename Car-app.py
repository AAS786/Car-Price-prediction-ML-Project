import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
import os
import warnings
warnings.filterwarnings("ignore")



# Load the model
model = pk.load(open('C:/Users/dell/Downloads/Car_Sales.sav', 'rb'))


# Load the car details data
cars_data = pd.read_csv('Cardetails.csv')

# Extract car brand names
def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()
cars_data['name'] = cars_data['name'].apply(get_brand_name)

# CSS to set the background image
page_bg_img = '''
<style>
body {
background-image: url("https://i.postimg.cc/wvN8crTn/1-ftn-M93-Qhl-S0-A7-I55-Qegbr-A.jpg");
background-size: cover;
}
</style>
'''

# Apply the background image
st.markdown(page_bg_img, unsafe_allow_html=True)

# Define the main function for the Streamlit app
def main():
    st.markdown("<h1 style='text-align: center; color: #4B0082;'>🚗Car Price Prediction ML Model🚗</h1>", unsafe_allow_html=True)
    
    # Collect user inputs
    col1, col2 = st.columns(2)
    with col1:
        name = st.selectbox('**Select Car Brand**', cars_data['name'].unique())
    with col2:
        year = st.slider('**Car Manufactured Year**', 1994, 2024)
    with col1:
        km_driven = st.slider('**No of kms Driven**', 11, 200000)
    with col2:
        fuel = st.selectbox('**Fuel type**', cars_data['fuel'].unique())
    with col1:
        seller_type = st.selectbox('**Seller Type**', cars_data['seller_type'].unique())
    with col2:
        transmission = st.selectbox('**Transmission Type**', cars_data['transmission'].unique())
    with col1:
        owner = st.selectbox('**Owner Type**', cars_data['owner'].unique())
    with col2:
        mileage = st.slider('**Car Mileage**', 10, 40)
    with col1:
        engine = st.slider('**Engine CC**', 700, 5000)
    with col2:
        max_power = st.slider('**Max Power**', 0, 200)
    #with col1:
    seats = st.slider('**No of Seats**', 5, 10)
    
    # Predict the car price
    if st.button("🚗 Predict 🚗"):
        input_data_model = pd.DataFrame(
            [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
            columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
        )
        input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
                                          [1, 2, 3, 4, 5], inplace=True)
        input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
        input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
        input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
        input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                                          'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                                          'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                                          'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                                          'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                                         inplace=True)
        car_price = model.predict(input_data_model)
        
        price_in_lakhs = car_price[0] / 100000
        price_in_lakhs_str = f"{price_in_lakhs:,.2f}"
        st.markdown(f'<h2 style="color: #32CD32;">💸 Car Price is going to be ₹{price_in_lakhs_str} Lakhs 💸</h2>', unsafe_allow_html=True)
        
    
if __name__ == '__main__':
    main()
