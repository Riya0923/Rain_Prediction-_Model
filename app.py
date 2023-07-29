from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import datetime as dt
# Create flask app
app = Flask(__name__)


pkl_filename = "logreg.pkl"
model = pickle.load(open(pkl_filename, "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    print(pd.__version__)
    print(np.__version__)
    Date = request.form.get('Date')
    Location = request.form.get('Location')
    MinTemp = request.form.get('MinTemp')
    MaxTemp = request.form.get('MaxTemp')
    Rainfall = request.form.get('Rainfall')
    Evaporation = request.form.get('Evaporation')
    Sunshine = request.form.get('Sunshine')
    WindGustDir = request.form.get('WindGustDir')
    WindGustSpeed = request.form.get('WindGustSpeed')
    WindDir9am = request.form.get('WindDir9am')
    WindDir3pm = request.form.get('WindDir3pm')
    WindSpeed9am = request.form.get('WindSpeed9am')
    WindSpeed3pm = request.form.get('WindSpeed3pm')
    Humidity9am = request.form.get('Humidity9am')
    Humidity3pm = request.form.get('Humidity3pm')
    Pressure9am = request.form.get('Pressure9am')
    Pressure3pm = request.form.get('Pressure3pm')
    Cloud9am = request.form.get('Cloud9am')
    Cloud3pm = request.form.get('Cloud3pm')
    Temp9am = request.form.get('Temp9am')
    Temp3pm = request.form.get('Temp3pm')
    RainToday = request.form.get('RainToday')
    
    
    
    column_list = ['Date','Location_Adelaide','Location_Albany','Location_Albury','Location_AliceSprings','Location_BadgerysCreek','Location_Ballarat','Location_Bendigo','Location_Brisbane','Location_Cairns','Location_Canberra','Location_Cobar','Location_CoffsHarbour','Location_Dartmoor','Location_Darwin','Location_GoldCoast','Location_Hobart','Location_Katherine','Location_Launceston','Location_Melbourne','Location_MelbourneAirport','Location_Mildura','Location_Moree','Location_MountGambier','Location_MountGinini','Location_Newcastle','Location_Nhil','Location_NorahHead','Location_NorfolkIsland','Location_Nuriootpa','Location_PearceRAAF','Location_Penrith','Location_Perth','Location_PerthAirport','Location_Portland','Location_Richmond','Location_Sale','Location_SalmonGums','Location_Sydney','Location_SydneyAirport','Location_Townsville','Location_Tuggeranong','Location_Uluru','Location_WaggaWagga','Location_Walpole','Location_Watsonia','Location_Williamtown','Location_Witchcliffe','Location_Wollongong','Location_Woomera', 'MinTemp',   'MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir_E','WindGustDir_ENE','WindGustDir_ESE','WindGustDir_N','WindGustDir_NE','WindGustDir_NNE','WindGustDir_NNW','WindGustDir_NW','WindGustDir_S','WindGustDir_SE','WindGustDir_SSE','WindGustDir_SSW','WindGustDir_SW','WindGustDir_W','WindGustDir_WNW','WindGustDir_WSW','WindGustSpeed','WindDir9am_E','WindDir9am_ENE','WindDir9am_ESE','WindDir9am_N','WindDir9am_NE','WindDir9am_NNE','WindDir9am_NNW','WindDir9am_NW','WindDir9am_S','WindDir9am_SE','WindDir9am_SSE','WindDir9am_SSW','WindDir9am_SW','WindDir9am_W','WindDir9am_WNW','WindDir9am_WSW','WindDir3pm_E','WindDir3pm_ENE','WindDir3pm_ESE','WindDir3pm_N','WindDir3pm_NE','WindDir3pm_NNE','WindDir3pm_NNW','WindDir3pm_NW','WindDir3pm_S','WindDir3pm_SE','WindDir3pm_SSE','WindDir3pm_SSW','WindDir3pm_SW','WindDir3pm_W','WindDir3pm_WNW','WindDir3pm_WSW','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','Year','Month','Day','RainToday']

    # Data Formation
    df = pd.DataFrame(0, index=np.arange(1), columns=column_list)

    # Numaric value update
    df['Date'] = Date
    df['RainToday'] = RainToday
    df['MinTemp'] = MinTemp 
    df['MaxTemp']= MaxTemp
    df['Rainfall'] = Rainfall
    df['Evaporation'] = Evaporation
    df['Sunshine'] = Sunshine
    df['WindGustSpeed'] = WindGustSpeed
    df['WindSpeed9am'] = WindSpeed9am
    df['WindSpeed3pm'] = WindSpeed3pm
    df['Humidity9am'] = Humidity9am
    df['Humidity3pm'] = Humidity3pm
    df['Pressure9am'] = Pressure9am
    df['Pressure3pm'] = Pressure3pm
    df['Cloud9am'] = Cloud9am
    df['Cloud3pm'] = Cloud3pm
    df['Temp9am'] = Temp9am
    df['Temp3pm'] = Temp3pm
    
    


    # Categorical value update
    df["WindGustDir" + "_" + WindGustDir] = 1
    df["WindDir9am" + "_" + WindDir9am] = 1
    df["WindDir3pm" + "_" + WindDir3pm] = 1
    df["Location" + "_" + Location] = 1
    df['RainToday'] = df['RainToday'].astype(str)
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df = df.drop(['Date'], axis='columns')
    #return render_template("index.html", prediction_text=df[df.columns[-1]])
    y_pred_test = model.predict(df)
    return render_template("index.html", prediction_text="Tomorrow's Rain Prediction is {}".format(y_pred_test))

if __name__ == "__main__":
    app.run(debug=True)