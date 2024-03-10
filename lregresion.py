import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load your data (replace 'your_data.csv' with your actual file path)
data = pd.read_csv('yield_d.csv')
data.dropna(inplace=True)
data = data[data['yield']>0]

# Feature selection
features = ['temperature', 'rainfall']  # Adjust as needed
target = 'yield'

# One-hot encoding for categorical variable (crop)
crops = data['crop'].unique()  # Get unique crop types
for crop in crops:
    data[f'crop_{crop}'] = (data['crop'] == crop).astype(int)
features.extend([f'crop_{crop}' for crop in crops])
#data.drop('crop', axis=1, inplace=True)  # Remove original crop column

# Data preprocessing (handle missing values, outliers, etc.)
# Example: handle missing values with imputation (e.g., mean/median)
#data = data.fillna(data.mean())  # Replace with appropriate imputation strategy

# Split data into training and testing sets stratified by crop
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42, stratify=data['crop'])

# Define and train a Random Forest model (consider other algorithms)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation (consider using crop-specific metrics)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")  # Consider crop-specific metrics later



# Assuming your model is named 'model'
# Save the model to a file
joblib.dump(model, 'my_model.pkl')  

model10 = joblib.load('my_model.pkl')

# Prediction on new data (optional)
# New data should include crop information and values for other features
new_data = pd.DataFrame({
    'temperature': [26.3],
    'rainfall': [1292],
    'crop_Rice': [0],  # Replace with 1 for crop A, 0 for others (one-hot encoding)
    'crop_Wheat': [0],  # Replace accordingly for other crop types
    'crop_Maize': [0], 
    'crop_Potatoes': [0], 
    'crop_Sorghum': [0], 
    'crop_Soybeans': [0], 
    'crop_Cassava': [0], 
    'crop_Spotatoes': [0],
    'crop_Plantains': [0], 
    'crop_Yams': [1]
    # ... add columns for other crops in your data
})
joblib.dump(features, 'my_features.pkl')  
nfeTURES = joblib.load('my_features.pkl')

predicted_yield = model10.predict(new_data[nfeTURES])[0]  # Access the first element

print(f"Predicted yield for new data ({new_data['crop_Yams'].values[0]}): {predicted_yield:.2f}")