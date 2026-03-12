from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import requests

app = Flask(__name__)

# -------- WEATHER API CONFIG --------
WEATHER_API_KEY = "YOUR_API_KEY"

# -------- Load Model & Encoders --------
model = joblib.load("fertilizer_model.pkl")
soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
fertilizer_encoder = joblib.load("fertilizer_encoder.pkl")

# -------- Fertilizer Dosage Logic --------
fertilizer_dosage = {
    "Urea": "45–55 kg per acre",
    "DAP": "50–60 kg per acre",
    "14-35-14": "40–50 kg per acre",
    "28-28": "35–45 kg per acre",
    "17-17-17": "30–40 kg per acre",
    "20-20": "40–50 kg per acre"
}

fertilizer_tips = {
    "Urea": "Apply in split doses. Avoid application before rainfall.",
    "DAP": "Best applied at sowing time.",
    "14-35-14": "Good for early root development.",
    "28-28": "Improves plant strength and growth.",
    "17-17-17": "Balanced fertilizer for all stages.",
    "20-20": "Ideal for vegetative growth."
}

# -------- WEATHER API ROUTE --------
@app.route("/weather")
def get_weather():
    city = request.args.get("city")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"

    response = requests.get(url).json()

    if response.get("main"):
        return jsonify({
    "temperature": int(round(response["main"]["temp"])),
    "humidity": int(round(response["main"]["humidity"]))
 })

    else:
        return jsonify({"error": "City not found"}), 404


# -------- MAIN ROUTE --------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = dosage = tip = None

    if request.method == "POST":
        try:
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            moisture = float(request.form["moisture"])
            nitrogen = float(request.form["nitrogen"])
            potassium = float(request.form["potassium"])
            phosphorous = float(request.form["phosphorous"])
            soil_type = request.form["soil"]
            crop_type = request.form["crop"]

            soil_encoded = soil_encoder.transform([soil_type])[0]
            crop_encoded = crop_encoder.transform([crop_type])[0]

            input_data = pd.DataFrame([[
                temperature, humidity, moisture,
                soil_encoded, crop_encoded,
                nitrogen, potassium, phosphorous
            ]], columns=model.feature_names_in_)

            pred = model.predict(input_data)
            prediction = fertilizer_encoder.inverse_transform(pred)[0]

            dosage = fertilizer_dosage.get(prediction)
            tip = fertilizer_tips.get(prediction)

        except Exception as e:
            prediction = "Error"
            dosage = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        dosage=dosage,
        tip=tip,
        soils=soil_encoder.classes_,
        crops=crop_encoder.classes_
    )


if __name__ == "__main__":
    app.run(debug=True)
