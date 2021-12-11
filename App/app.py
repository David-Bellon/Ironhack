import pandas as pd
import numpy as np
import pickle
import streamlit as st

df = pd.read_excel(r"Data\regression_data.xls")

price_per_zip = {}

zips = df["zipcode"]
prices = df["price"]
for i in range(len(zips)):
    if zips[i] in price_per_zip.keys():
        price_per_zip[zips[i]].append(prices[i])
    else:
        price_per_zip[zips[i]] = []
        price_per_zip[zips[i]].append(prices[i])

mean_price_per_zip = {}
for key in price_per_zip.keys():
    mean_price_per_zip[key] = np.mean(price_per_zip[key])



grboos = pickle.load(open("Models\gradient_boosting_model.pkl", "rb"))
linear = pickle.load(open("Models\linear_regresion_model.pkl", "rb"))

scaler = pickle.load(open("Scalers\scaler.pkl", "rb"))

st.write("House Price Prediction")
st.markdown('<style>body{background-color: Black}</style>',unsafe_allow_html=True)


bedrooms = st.number_input("Enter the number of bedrooms")
bathrooms = st.number_input("Enter the number of bathrooms")
sqft_living = st.number_input("Enter the square foot of the house")
sqft_lot = st.number_input("Enter the square foot of the land")
floors = st.number_input("Enter the number of floors of the house")
waterfront = st.number_input("Enter 1 if the house has views to water, 0 if not")
view = st.number_input("Enter the grade of the view from the house between 0 to 4")
condition = st.number_input("Enter the condition of the house in range between 1 to 5")
grade = st.number_input("Enter the grade of the house between 1 to 13")
sqft_above = st.number_input("Enter the square foot of the higher floors")
sqft_basement = st.number_input("Enter the square foot of the basement in case it has")
year_built = st.number_input("Enter the built year of the house")
year_renovated = st.number_input("Enter the year the house was renovated if it is")
zipcode = st.number_input("Enter the zipcode of the house")
lat = st.number_input("Enter the latitud of the house")
long = st.number_input("Enter the longitud of the house")
sqft_living15 = st.number_input("Enter the living square foot in 2015")
sqft_lot15 = st.number_input("Enter the land square foot in 2015")


if st.button("Get the value of the house"):
    x = pd.DataFrame({"bedrooms": [bedrooms], "bathrooms": [bathrooms], "sqft_living": [sqft_living], "sqft_lot": [sqft_lot], "floors": [floors], "waterfront": [waterfront],
    "view": [view], "condition": [condition], "grade": [grade], "sqft_above": [sqft_above], "sqft_basement": [sqft_basement], "year_built": [year_built], "year_renovated": [year_renovated],
    "zipcode": [zipcode], "lat": [lat], "long": [long], "sqft_living15": [sqft_living15], "sqft_lot15": [sqft_lot15], "mean_price_per_zipcode": [mean_price_per_zip[zipcode]]})


    #Scale Data
    #X_scaled = scaler.transform(x)
    #df_scaled = pd.DataFrame(X_scaled, columns=x.columns)

    #Predicts
    prediction = grboos.predict(x)

    final_text = "The stimated value of the house is" + str(prediction)
    st.success(final_text)
