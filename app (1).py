from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))
fertilizer_model = pickle.load(open('fertilizer_model.pkl', 'rb'))

with open('fertilizer_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('soil_encoder (1).pkl', 'rb') as soil_file:
    loaded_soil_encoder = pickle.load(soil_file)

with open('crop_encoder (1).pkl', 'rb') as crop_file:
    loaded_crop_encoder = pickle.load(crop_file)

with open('fertilizer_encoder (1).pkl', 'rb') as ferti_file:
    loaded_ferti_encoder = pickle.load(ferti_file)





# creating flask app
app = Flask(__name__)

@app.route('/')
def main_home():
    return render_template("home.html")

@app.route("/crop_recommendation")
def crop_recommendation():
    return render_template('index.html') 

@app.route("/crop_yield")
def crop_yield():
    return render_template('index1.html') 

@app.route("/fertilizer_recommendation")
def fertilizer_recommendation():
    return render_template('fertilizer.html') 

@app.route("/home")
def home():
    return render_template('home.html')


@app.route("/predict1",methods=['POST'])
def predict1():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result = result)


@app.route("/predict2",methods=['POST'])
def predict2():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item  = request.form['Item']

        features = np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]],dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1,-1)

        return render_template('index1.html',prediction = prediction)
    

@app.route('/predict3', methods=[ 'POST'])
def predict3():
    if request.method == 'POST':
        Temperature= float(request.form['Temperature'])
        Humidity= float(request.form['Humidity'])
        Moisture= float(request.form['Moisture'])
        Soil_Type=request.form['soil_type']

        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Pottasium'])
        Crop_Type = request.form['crop_type']

        encoded_soil = loaded_soil_encoder.transform([Soil_Type])[0]
        encoded_crop = loaded_crop_encoder.transform([Crop_Type])[0]


        # Encode or map crop_type if needed for model input
        features1 = np.array([[Temperature, Humidity, Moisture, encoded_crop,encoded_soil,N, P, K]])  # or include crop_type if applicable
        prediction1 = loaded_model.predict(features1)[0]
        decoded_prediction= loaded_ferti_encoder.inverse_transform([prediction1])[0]
        

        return render_template('fertilizer.html', prediction1= decoded_prediction)
    
    






# python main
if __name__ == "__main__":
    app.run(port=5002,debug=True)