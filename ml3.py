import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


APP_FOLDER = Path(__file__).parent


st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)


## LOAD MODEL ##
@st.cache_resource
def load_pipe():
    return joblib.load(APP_FOLDER / "model.pkl")


pipe = load_pipe()


## LOAD DATA ##
df = pd.read_csv(APP_FOLDER / "data.csv")


## INPUT FROM USER ##
st.sidebar.header("Enter Car Information")

make = st.sidebar.selectbox(
    "Make",
    sorted(df["Make"].dropna().unique())
)

models = df[df["Make"] == make]["Model"].dropna().unique()

model = st.sidebar.selectbox(
    "Model",
    sorted(models)
)

year = st.sidebar.slider(
    "Year",
    min_value=int(df["Year"].min()),
    max_value=int(df["Year"].max()),
    value=int(df["Year"].max())
)

engine_fuel_type = st.sidebar.selectbox(
    "Engine Fuel Type",
    sorted(df["Engine Fuel Type"].dropna().unique())
)

engine_hp = st.sidebar.number_input(
    "Engine HP",
    min_value=1.0,
    value=200.0
)

engine_cylinders = st.sidebar.number_input(
    "Engine Cylinders",
    min_value=0.0,
    value=4.0
)

transmission_type = st.sidebar.selectbox(
    "Transmission Type",
    sorted(df["Transmission Type"].dropna().unique())
)

driven_wheels = st.sidebar.selectbox(
    "Driven Wheels",
    sorted(df["Driven_Wheels"].dropna().unique())
)

number_of_doors = st.sidebar.number_input(
    "Number of Doors",
    min_value=2.0,
    max_value=4.0,
    value=4.0
)

market_category = st.sidebar.selectbox(
    "Market Category",
    sorted(df["Market Category"].dropna().unique())
)

vehicle_size = st.sidebar.selectbox(
    "Vehicle Size",
    sorted(df["Vehicle Size"].dropna().unique())
)

vehicle_style = st.sidebar.selectbox(
    "Vehicle Style",
    sorted(df["Vehicle Style"].dropna().unique())
)

highway_mpg = st.sidebar.number_input(
    "Highway MPG",
    min_value=1,
    value=30
)

city_mpg = st.sidebar.number_input(
    "City MPG",
    min_value=1,
    value=20
)

popularity = st.sidebar.number_input(
    "Popularity",
    min_value=1,
    value=1000
)


## PREPARE DATA FOR MODEL ##
latest_year = int(df["Year"].max())
vehicle_age = latest_year - year
average_mpg = (highway_mpg + city_mpg) / 2

new_data = {
    "Make": make,
    "Model": model,
    "Year": year,
    "Engine Fuel Type": engine_fuel_type,
    "Engine HP": engine_hp,
    "Engine Cylinders": engine_cylinders,
    "Transmission Type": transmission_type,
    "Driven_Wheels": driven_wheels,
    "Number of Doors": number_of_doors,
    "Market Category": market_category,
    "Vehicle Size": vehicle_size,
    "Vehicle Style": vehicle_style,
    "highway MPG": highway_mpg,
    "city mpg": city_mpg,
    "Popularity": popularity,
    "Vehicle Age": vehicle_age,
    "Average MPG": average_mpg
}

new_data_df = pd.DataFrame(new_data, index=[0])


## PREDICTION ##
st.title("Car MSRP Prediction")
st.write("Enter the car information in the sidebar, then press Predict.")

button = st.button("Predict")

if button:
    result = pipe.predict(new_data_df)
    st.write("Predicted MSRP:")
    st.write("$", round(result[0], 2))
