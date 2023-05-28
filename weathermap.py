import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style='darkgrid')
import streamlit as st
import pyttsx3
import speech_recognition as sr
import time
import os
import torch
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

# ------------------------------------------ GET DATA -------------------

# %pip install meteostat --q
from meteostat import Stations, Daily
# Using the Meteostat API, we need the station ID of each state in Nigeria and the data start and ending dates

# Get the station IDs for each state in Nigeria
# Connect the API with the station ID and the start/end date
# Use 2021 as start date and today's date as end date

from datetime import datetime
current_date = datetime.today().strftime('%Y-%m-%d')

def get_data(state):
    station = Stations()
    nig_stations = station.region('NG')  # Filter stations by country code (NG for Nigeria)
    nig_stations = nig_stations.fetch()  # Fetch the station information
    global available_states
    # Some state names have a '/' in them. So we clean them up
    nig_stations['name'] = nig_stations['name'].apply(lambda x: x.split('/', 1)[0])
    nig_stations.drop_duplicates(subset=['name'], keep='first', inplace=True)
    available_states = nig_stations.name
    # Collect the data from the mentioned state if state is in the list of available states
    try:
        state_stations = nig_stations[nig_stations['name'].str.contains(state)]
        station_id = state_stations.index[0]
    except IndexError:
        error = f'Sorry, {state} is not among the available states. Please choose another neighboring state'
        raise ValueError(error)

    # Connect the API and fetch the data
    data = Daily(station_id, str(state_stations.hourly_start[0]).split(' ')[0], str(current_date))
    data = data.fetch()

    # Collect the necessary features we might need
    data['avg_temp'] = data[['tmin', 'tmax']].mean(axis=1)
    temp = data['avg_temp']
    press = data['pres']
    wind_speed = data['wspd']
    precip = data['prcp']
    rain_df = pd.concat([temp, press, wind_speed, precip], axis=1)

    # From the collected data, create a DataFrame for training the model
    # Light rain — when the precipitation rate is < 2.5 mm (0.098 in) per hour.
    # Moderate rain — when the precipitation rate is between 2.5 mm (0.098 in) – 7.6 mm (0.30 in) or 10 mm (0.39 in) per hour.
    # Heavy rain — when the precipitation rate is > 7.6 mm (0.30 in) per hour, or between 10 mm (0.39 in) and 50 mm (2.0 in)
    rainfall = []
    for i in rain_df.prcp:
        if i < 0.1:
            rainfall.append('no rain')
        elif i < 2.5:
            rainfall.append('light rain')
        elif i > 2.5 and i < 7.6:
            rainfall.append('moderate rain')
        else: 
            rainfall.append('heavy rain')
    model_data = rain_df.copy()
    rainfall = pd.Series(rainfall)
    model_data.reset_index(inplace=True)
    model_data['raining'] = rainfall
    model_data.index = model_data['time']
    model_data.drop(['time'], axis=1, inplace=True)
    model_data.dropna(inplace=True)

    return data, temp, press, wind_speed, precip, model_data

  
# ------------------------------GET States -----------------

nigerian_states = ['Sokoto', 'Gusau', 'Kaduna', 'Zaria', 'Kano', 'Maiduguri',
                   'Ilorin', 'Bida', 'Minna', 'Abuja', 'Jos', 'Yola', 'Lagos',
                   'Ibadan', 'Oshogbo', 'Benin City', 'Port Harcourt', 'Enugu',
                   'Calabar', 'Makurdi', 'Uyo', 'Akure', 'Imo', 'Minna']

# ------------------------- Modelling For Precipitation (Rainfall) ---------------------------

def modelling(data, target):
    # Preprocess the data
    scaler = StandardScaler()
    encode = LabelEncoder()
    x = data.drop(target, axis=1)
    for i in x.columns:
        x[[i]] = scaler.fit_transform(x[[i]])
    y = encode.fit_transform(data[target])

    # Find the best random state
    model_score = []
    for i in range(1, 100):
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, stratify=y, random_state=i)
        xgb_model = XGBClassifier()
        xgb_model.fit(xtrain, ytrain)
        prediction = xgb_model.predict(xtest)
        score = accuracy_score(ytest, prediction)
        model_score.append(score)

    best_random_state = [max(model_score), model_score.index(max(model_score))]

    # Now we start training and since we have gotten the best random state
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=best_random_state[1] + 1)
    xgb_model = XGBClassifier()
    xgb_model.fit(xtrain, ytrain)
    prediction = xgb_model.predict(xtest)
    score = accuracy_score(ytest, prediction)
    report = classification_report(ytest, prediction)
    report = classification_report(ytest, prediction, output_dict=True)
    report = pd.DataFrame(report).transpose()
    print(report)
    print(f"Accuracy Score: {score}")
    joblib.dump(xgb_model, 'precipitation_model.pkl')

    return xgb_model, best_random_state, report

# Example usage
data, temp, press, wind_speed, precip, model_data = get_data('Lagos')
xgb_model, best_random_state, report = modelling(model_data, 'raining')



# ............... ARIMA TIME SERIES FOR TEMPERATURE, PRESSURE, AND WIND-SPEED ............

# Get time and response variable of the original data
def response_data(df):
    pressure = df['pres']
    temp = df['avg_temp']
    wind = df['wspd']
    precip = df['prcp']
    return pressure, temp, wind, precip

# Plotter
def plotter(dataframe):
    plt.figure(figsize=(15, 3))
    plt.subplot(1, 1, 1)
    sns.lineplot(data=dataframe)

# Date Collector
def item_collector(data, selected_date):
    selected_row = data.loc[data.index == selected_date]
    return float(selected_row.values[0])

# Determining lag value for our time series model by looping through possible numbers using GRID SEARCH method
def best_parameter(data):
    # Create a grid search of possible values of p, d, and q values
    p_values = range(0, 5)
    d_values = range(0, 2)
    q_values = range(0, 4)

    # Create a list to store the best AIC values and the corresponding p, d, and q values
    best_aic = np.inf
    best_pdq = None

    # Loop through all possible combinations of p, d, and q values
    for p in p_values:
        for d in d_values:
            for q in q_values:
                # Fit the ARIMA model
                model = sm.tsa.ARIMA(data, order=(p, d, q))
                try:
                    model_fit = model.fit()
                    # Update the best AIC value and the corresponding p, d, and q values if the current AIC value is lower
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_pdq = (p, d, q)
                except:
                    continue
    return best_pdq


# --------- MODELLING FOR TIME SERIES USING ARIMA MODEL -------------------
# Create a function that models, predicts, and returns the model and the predicted values
def arima_model(data, best_param):
    model = sm.tsa.ARIMA(data, order=best_param)
    model = model.fit()

    # Plot the Future Predictions according to the model
    future_dates = pd.date_range(start=data.index[-1], periods=5, freq='D')
    forecast = model.predict(start=len(data), end=len(data) + (5 - 1))

    # Get the dataframes of the original/prediction
    data_ = data.copy()
    data_['predicted'] = model.fittedvalues

    # Get the dataframe for predicted values
    predictions_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
    predictions_df.set_index('Date', drop=True, inplace=True)
    joblib.dump(model, 'time_series_model.pkl')

    return predictions_df

# Example usage
data, temp, press, wind_speed, precip, model_data = get_data('Lagos')
best_param = best_parameter(temp)
predictions = arima_model(temp, best_param)

#  --------------- EMBEDDING TALKBACK ---------------
# Create a function for text Talkback
def Text_Speaker(your_command):
    speaker_engine = pyttsx3.init() #................ initiate the talkback engine
    speaker_engine.say(your_command) #............... Speak the command
    speaker_engine.runAndWait() #.................... Run the engine

instructions = "Follow the instructions below to effectively use the app: 1. Launch the App: Open the Weather Prediction App on your device. 2.Enable Voice Input: Make sure your devices microphone is enabled and properly working. Grant necessary permissions if prompted. 3. Speak Naturally: Communicate with the app using natural language and conversational tone. You can ask questions or provide instructions using voice commands. 4. Mention Weather-related Keywords: When you want to inquire about the weather or climate, use words like 'weather' or 'climate' in your conversation. For example: Whats the weather like today?, Tell me the climate forecast for tomorrow., Is it going to rain in London?. 5. Provide Location Information: Specify the location for which you want the weather prediction. You can mention the city name, ZIP code, or any other relevant location details. 6. Listen to the Response: The app will process your voice command, retrieve the necessary weather information, and provide a response. You can listen to the apps voice output or read the displayed information on your devices screen. 7. End the Conversation: When you have received the desired weather prediction or want to exit the app, you can end the conversation by saying 'Goodbye', 'Thank you', or any other appropriate closing remark. Enjoy Your Usage !!!"


# --------------------- EMBEDDING SPEECH RECOGNITION ------------------------------ 
# Create a function for Speech Trancription
def transcribe_speech():
    # Initialize recognizer class
    r = sr.Recognizer()

    # Reading Microphone as source
    with sr.Microphone() as source:

        # create a streamlit spinner that shows progress
        with st.spinner(text='Silence pls, Caliberating background noise.....'):
            time.sleep(3)

        r.adjust_for_ambient_noise(source, duration = 1) # ..... Adjust the sorround noise
        st.info("Speak now...")

        audio_text = r.listen(source) #........................ listen for speech and store in audio_text variable
        with st.spinner(text='Transcribing your voice to text'):
            time.sleep(2)

        try:
            # using Google speech recognition to recognise the audio
            text = r.recognize_google(audio_text)
            # print(f' Did you say {text} ?')
            return text
        except:
            return "Sorry, I did not get that."
        

#----------------streamlit implementation---------------------------#

st.title('Weather Predictor')
st.markdown('My name is Louis! I will be predicting your weather today.')
st.image('weather.png', caption='Source: Unsplash')

username = st.text_input('What is your name?', key='name')
button = st.button('Please click me to submit.')

if button:
    if username != '':
        speak(f'Hello, {username}')
        st.markdown(f'Hello, {username}!')
        speak('Please select a date and location to get your weather prediction...')
    else:
        speak('Please enter your name...')
        st.warning('Please input your username to continue.')

st.sidebar.subheader('Weather Prediction')
prediction_mode = st.sidebar.selectbox('Select prediction mode:', ['Text', 'Voice'])

if prediction_mode == 'Text':
    st.subheader('Weather Prediction (Text)')
    place = st.text_input('Enter location:')
    date_string = st.text_input('Enter date (YYYY-MM-DD):')

    if st.button('Predict'):
        # Check if the user provided a location
        if not place:
            st.warning('Please enter a location.')
            # No need for 'return' here

        # Check if the user provided a date
        if not date_string:
            date_string = datetime.today().strftime('%Y-%m-%d')
            st.warning(f"No date provided. Using today's date: {date_string}")

        # Get the weather data for the specified location
        data, temp, press, wind_speed, precip, model_data = get_data(place)

        # ------------ Get response data for time series
        pressure_data, temperature_data, wind_data, precipitation_data = response_data(data, press, temp, wind_speed, precip)
        # ------------ Get Best Time Series Parameters
        pressure_param = best_parameter(pressure_data)
        wind_param = best_parameter(wind_data)
        temp_param = best_parameter(temperature_data)
        precip_param = best_parameter(precipitation_data)
        # ------------ Time Series Prediction
        pressure_predict = arima_model(pressure_data, pressure_param)
        windSpeed_predict = arima_model(wind_data, wind_param)
        temperature_predict = arima_model(temperature_data, temp_param)
        precipitation_predict = arima_model(precipitation_data, precip_param)
        # ------------ Collect the future data for the specified date
        if date_string in pressure_predict.index:
            freq = 'D'  # Adjust the frequency as per your data
            future_pressure = pd.Series(item_collector(pressure_predict, date_string), index=pd.DatetimeIndex([date_string], freq=freq))
            future_windSpeed = pd.Series(item_collector(windSpeed_predict, date_string), index=pd.DatetimeIndex([date_string], freq=freq))
            future_temperature = pd.Series(item_collector(temperature_predict, date_string), index=pd.DatetimeIndex([date_string], freq=freq))
            future_precipitation = pd.Series(item_collector(precipitation_predict, date_string), index=pd.DatetimeIndex([date_string], freq=freq))
        else:
            st.warning('Oops, the date you chose is beyond 5 days. Please choose a date within the next 5 days.')

        if 'rainfall' in your_words_in_text.lower().strip():
            print(f"You mentioned to predict rainfall for {date_string}. Please wait while I get to work...")
            # ------------- Modelling
            xgb_model, best_random_state, report = modelling(model_data, 'raining')
            # Testing the Rainfall Model (In order of Temp, Pressure, Windspeed, Precipitation)
            input_value = [[future_temperature, future_pressure, future_windSpeed, future_precipitation]]
            scaled = StandardScaler().fit_transform(input_value)
            prediction = xgb_model.predict(scaled)
            if int(prediction) == 0:
                st.success('There won\'t be rain.')
            elif int(prediction) == 1:
                st.success('There will be light rain.')
            elif int(prediction) == 2:
                st.success('There will be a moderate level of rain.')
            else:
                st.success('There will be heavy rain.')

        elif 'temperature' in your_words_in_text.lower().strip():
            st.success(f"You mentioned to predict temperature for {date_string}. Please wait while I get to work...")
            st.success(f"The temperature for {place} at {date_string} is {future_temperature}")

        elif 'wind speed' in your_words_in_text.lower().strip():
            st.success(f"You mentioned to predict wind speed for {date_string}. Please wait while I get to work...")
            st.success(f"The wind speed for {place} at {date_string} is {future_windSpeed}")

        elif 'pressure' in your_words_in_text.lower().strip():
            st.success(f"You mentioned to predict pressure for {date_string}. Please wait while I get to work...")
            st.success(f"The pressure for {place} at {date_string} is {future_pressure}")

        elif 'precipitation' in your_words_in_text.lower().strip():
            st.success(f"You mentioned to predict precipitation for {date_string}. Please wait while I get to work...")
            st.success(f"The precipitation for {place} at {date_string} is {future_precipitation}")

        else:
            st.error('Could not find an action.')

st.warning('Please note that I will automatically select Ibadan in the absence of a location.')

if st.button("Communicate with me"):
    your_words_in_text = transcribe_speech()
    st.write("Transcription: ", your_words_in_text)
    place = [i for i in your_words_in_text.capitalize().split() if i in nigerian_states]
    # Set the default value to 'Ibadan' if no match is found
    if not place:
        place = 'Ibadan'
    place_data, temp, press, wind_speed, precip, model_data = get_data(place[0])  # Call the get_data function to fetch place data.

    selected_date = st.date_input("Select a date. Remember it shouldn't be more than 5 days away from now")
    # Collect the date
    date_string = selected_date.strftime("%Y-%m-%d")  # Turn the date to a string

    # Get response data for time series
    pressure_data, temperature_data, wind_data, precipitation_data = response_data(place_data, press, temp, wind_speed, precip)
    # Get Best Time Series Parameters
    pressure_param = best_parameter(pressure_data)
    wind_param = best_parameter(wind_data)
    temp_param = best_parameter(temperature_data)
    precip_param = best_parameter(precipitation_data)
    # Time Series Prediction
    pressure_predict = arima_model(pressure_data, pressure_param)
    windSpeed_predict = arima_model(wind_data, wind_param)
    temperature_predict = arima_model(temperature_data, temp_param)
    precipitation_predict = arima_model(precipitation_data, precip_param)
    # Collect the future data for the specified date
    if date_string in pressure_predict.index:
        future_pressure = item_collector(pressure_predict, date_string)
        future_windSpeed = item_collector(windSpeed_predict, date_string)
        future_temperature = item_collector(temperature_predict, date_string)
        future_precipitation = item_collector(precipitation_predict, date_string)
    else:
        st.warning('Oops, the date you chose is beyond 5 days. Please choose a date within the next 5 days.')

    if 'rainfall' in your_words_in_text.lower().strip():
        print(f"You mentioned to predict rainfall for {date_string}. Please wait while I get to work...")
        # Modelling
        xgb_model, best_random_state, report = modelling(model_data, 'raining')
        # Testing the Rainfall Model (In order of Temp, Pressure, Windspeed, Precipitation)
        input_value = [[future_temperature, future_pressure, future_windSpeed, future_precipitation]]
        scaled = StandardScaler().fit_transform(input_value)
        prediction = xgb_model.predict(scaled)
        if int(prediction) == 0:
            st.success('There won\'t be rain.')
        elif int(prediction) == 1:
            st.success('There will be light rain.')
        elif int(prediction) == 2:
            st.success('There will be a moderate level of rain.')
        else:
            st.success('There will be heavy rain.')

    elif 'temperature' in your_words_in_text.lower().strip():
        st.success(f"You mentioned to predict temperature for {date_string}. Please wait while I get to work...")
        st.success(f"The temperature for {place} at {date_string} is {future_temperature}")

    elif 'wind speed' in your_words_in_text.lower().strip():
        st.success(f"You mentioned to predict wind speed for {date_string}. Please wait while I get to work...")
        st.success(f"The wind speed for {place} at {date_string} is {future_windSpeed}")

    elif 'pressure' in your_words_in_text.lower().strip():
        st.success(f"You mentioned to predict pressure for {date_string}. Please wait while I get to work...")
        st.success(f"The pressure for {place} at {date_string} is {future_pressure}")

    elif 'precipitation' in your_words_in_text.lower().strip():
        st.success(f"You mentioned to predict precipitation for {date_string}. Please wait while I get to work...")
        st.success(f"The precipitation for {place} at {date_string} is {future_precipitation}")

    else:
        st.error('Could not find an action.')

st.warning('Please note that I will automatically select Ibadan in the absence of a location.')