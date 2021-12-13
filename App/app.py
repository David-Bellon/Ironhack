from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from streamlit.util import index_


df = pd.read_excel(r"Data\regression_data.xls")

df = df.drop(columns=["id", "date"])

df["bathrooms"] = df["bathrooms"].astype(int)
df["floors"] = df["floors"].astype(int)

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

st.title("Bellon Alvareda SL")

st.write("Welcome to our application here you can select what you whant to do, you can run the house prediction app or search for houses in our database.")

page_selection = st.radio("Page", ("Prediction app", "Search houses"))

if page_selection == "Prediction app":
    st.title("House prediction App")
    st.write("You can select one of the models or both of them and see the differences")
    st.write("We recomend ussing the gradient boosting model due to its highest acuracy")

    model_selected = st.radio("Select Model", ("Linear Regression", "Gradient Boosting"))


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

        if model_selected == "Gradient Boosting":
            x = pd.DataFrame({"bedrooms": [bedrooms], "bathrooms": [bathrooms], "sqft_living": [sqft_living], "sqft_lot": [sqft_lot], "floors": [floors], "waterfront": [waterfront],
            "view": [view], "condition": [condition], "grade": [grade], "sqft_above": [sqft_above], "sqft_basement": [sqft_basement], "year_built": [year_built], "year_renovated": [year_renovated],
            "zipcode": [zipcode], "lat": [lat], "long": [long], "sqft_living15": [sqft_living15], "sqft_lot15": [sqft_lot15], "mean_price_per_zipcode": [mean_price_per_zip[zipcode]]})


            #Scale Data
            #X_scaled = scaler.transform(x)
            #df_scaled = pd.DataFrame(X_scaled, columns=x.columns)

            #Predicts
            prediction = grboos.predict(x)
            prediction = np.trunc(prediction)

            final_text = "The stimated value of the house by the gradient boosting model is $" +  str(prediction).replace("[", "").replace("]", "").replace(".", "")
            st.success(final_text)

        else:
            y = pd.DataFrame({"bedrooms": [bedrooms], "bathrooms": [bathrooms], "sqft_living": [sqft_living], "sqft_lot": [sqft_lot], "floors": [floors], "waterfront": [waterfront],
            "view": [view], "condition": [condition], "grade": [grade], "sqft_above": [sqft_above], "sqft_basement": [sqft_basement], "year_built": [year_built], "year_renovated": [year_renovated],
            "zipcode": [zipcode], "lat": [lat], "long": [long], "sqft_living15": [sqft_living15], "sqft_lot15": [sqft_lot15]})


            #Scale Data
            X_scaled = scaler.transform(y)
            df_scaled = pd.DataFrame(X_scaled, columns=y.columns)

            #Predicts
            prediction = np.exp(linear.predict(df_scaled))
            prediction = np.trunc(prediction)

            final_text = "The stimated value of the house by the linear regression model is $" + str(prediction).replace("[", "").replace("]", "").replace(".", "")
            st.success(final_text)
else:
    st.title("House search")
    st.write("Select the different aspects of the house that you want.")
    selection_bed = False
    with st.expander("Bedrooms. If non in selected it will count as not filtered"):
        if st.checkbox("All"):
            selection_bed = True
        bed1 = st.checkbox("1", selection_bed)
        bed2 = st.checkbox("2", selection_bed)
        bed3 = st.checkbox("3", selection_bed)
        bed4 = st.checkbox("4", selection_bed)
        bed5 = st.checkbox(">4", selection_bed)

    selection_bath = False   
    with st.expander("Bathrooms. If non in selected it will count as not filtered"):
        if st.checkbox("All", key = 0):
            selection_bath = True
        bath1 = st.checkbox("1", selection_bath, key = 1)
        bath2 = st.checkbox("2", selection_bath, key = 2)
        bath3 = st.checkbox("3", selection_bath, key = 3)
        bath4 = st.checkbox(">4", selection_bath, key = 4)

    number_floors = st.radio("Floors", (1, ">2"))
    
    waterviews = st.radio("Views to the water (Lake, see...)", ("Yes", "No"))
    
    condition = st.slider("Select the minimum condition of the house (from low to high)", 1, 5)

    grade = st.slider("The minimum grade of the house (Low to High)", 1, 13)

    price = st.slider("Max price of the house", 0, 10000000)

    #Get the number of bedroms

    beds = [bed5, bed4, bed3, bed2, bed1]

    if beds[0] == True:
        number_beds = df["bedrooms"].max()
    elif beds[1] == True:
        number_beds = 4
    elif beds[2] == True:
        number_beds = 3
    elif beds[3] == True:
        number_beds = 2
    elif beds[4] == True:
        number_beds = 1
    else:
        number_beds = df["bedrooms"].max()

    
    #Get the number of bathrooms
    
    baths = [bath4, bath3, bath2, bath1]

    if baths[0] == True:
        number_baths = df["bathrooms"].max()
    elif baths[1] == True:
        number_baths = 3
    elif baths[2] == True:
        number_baths = 2
    elif baths[0] == True:
        number_baths = 1
    else:
        number_baths = df["bathrooms"].max()


    #Get the floors
    if number_floors == ">2":
        number_floors = df["floors"].max()


    #Get the waterfront

    if waterviews == "Yes":
        water_view = 1
    else:
        water_view = 0



    if st.button("Show Houses"):

        see = df.loc[df["bedrooms"] <= number_beds]
        see = see.loc[see["bathrooms"] <= number_baths]
        see = see.loc[see["floors"] <= number_floors]
        see = see.loc[see["waterfront"] == water_view]
        see = see.loc[see["condition"] >= condition]
        see = see.loc[see["grade"] >= grade]
        see = see.loc[see["price"] <= price]

        if see.empty:
            st.error("No houses in that range of filters")
        else:
            st.success("These are the houses that match your requires")
            st.dataframe(see)

        see = see.reset_index().drop(columns=["index"])

        if see.shape[0] <= 10:
            for i in range(see.shape[0]):
                st.write("Description of the house")
                st.table(see.iloc[i])
                st.write("Location of the house")
                coord = pd.DataFrame({"lat": [see["lat"][i]], "lon": [see["long"][i]]})

                st.map(coord)
        else:
            coord = pd.DataFrame({"lat": see["lat"], "lon": see["long"]})
            st.map(coord)

        
        
        

            
    

        
        




    
