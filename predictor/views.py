# predictor/views.py
import os
import pandas as pd
import joblib
from django.shortcuts import render
from django.conf import settings

# Load model (only once)
model_path = os.path.join(settings.BASE_DIR, 'predictor', 'ML Model', 'rain_predict.pkl')
model = joblib.load(model_path)

def home(request):
    return render(request, 'predictor/home.html')

def predict_rain(request):
    prediction = None
    if request.method == 'POST':
        # Extract form data
        input_data = {
            'MinTemp': float(request.POST.get('MinTemp')),
            'MaxTemp': float(request.POST.get('MaxTemp')),
            'Rainfall': float(request.POST.get('Rainfall')),
            'Evaporation': float(request.POST.get('Evaporation')),
            'Sunshine': float(request.POST.get('Sunshine')),
            'WindGustSpeed': float(request.POST.get('WindGustSpeed')),
            'WindSpeed9am': float(request.POST.get('WindSpeed9am')),
            'WindSpeed3pm': float(request.POST.get('WindSpeed3pm')),
            'Humidity9am': float(request.POST.get('Humidity9am')),
            'Humidity3pm': float(request.POST.get('Humidity3pm')),
            'Pressure9am': float(request.POST.get('Pressure9am')),
            'Pressure3pm': float(request.POST.get('Pressure3pm')),
            'Cloud9am': float(request.POST.get('Cloud9am')),
            'Cloud3pm': float(request.POST.get('Cloud3pm')),
            'Temp9am': float(request.POST.get('Temp9am')),
            'Temp3pm': float(request.POST.get('Temp3pm')),
            'RainToday': int(request.POST.get('RainToday')),
        }

        # Convert to DataFrame and reorder
        df_input = pd.DataFrame([input_data])
        df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

        # Predict
        result = model.predict(df_input)[0]
        prediction = 'Rain' if result == 1 else 'No Rain'

    return render(request, 'predictor/predict.html', {'prediction': prediction})
