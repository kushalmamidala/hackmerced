from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # Your model
import joblib

app = Flask(__name__,template_folder='tmp')

# Load your model (replace with your model loading code)
model = joblib.load('my_model.pkl')
features = ['temperature', 'rainfall', 'crop_Maize', 'crop_Potatoes', 'crop_Rice', 'crop_Sorghum', 'crop_Soybeans', 'crop_Wheat', 'crop_Cassava', 'crop_Spotatoes', 'crop_Plantains', 'crop_Yams']
nfeTURES = joblib.load('my_features.pkl')

@app.route('/')
def home():
      return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temperature = float(request.form['temperature'])
    rainfall = float(request.form['rainfall'])
    crop = request.form['crop']

    # Create new_data DataFrame with one-hot encoding (adjust as needed)
   

    new_data = pd.DataFrame({
            'temperature': [temperature],
            'rainfall': [rainfall],
            'crop_Rice': [1 if crop == 'Rice' else 0],  # Replace with 1 for crop A, 0 for others (one-hot encoding)
            'crop_Wheat': [1 if crop == 'Wheat' else 0],  # Replace accordingly for other crop types
            'crop_Maize': [1 if crop == 'Maize' else 0], 
            'crop_Potatoes': [1 if crop == 'Potatoes' else 0], 
            'crop_Sorghum': [1 if crop == 'Sorghum' else 0], 
            'crop_Soybeans': [1 if crop == 'Soybeans' else 0], 
            'crop_Cassava': [1 if crop == 'Cassava' else 0], 
            'crop_Spotatoes': [1 if crop == 'Spotatoes' else 0],
            'crop_Plantains': [1 if crop == 'Plantains' else 0], 
            'crop_Yams': [1 if crop == 'Yams' else 0]
            # ... add columns for other crops (set to 0)
        })


    prediction = model.predict(new_data[nfeTURES])[0]  # Get the prediction

    return render_template('index.html', prediction=prediction)
     
     

if __name__ == '__main__':
    app.run(debug=True)