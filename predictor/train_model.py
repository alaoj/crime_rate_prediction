import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Load dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'global_crime_data.csv')

data = pd.read_csv(DATA_PATH)

# Select features and target
X = data[['Population', 'Unemployment_Rate', 'Education_Index']]
y = data['Crime_Rate']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
MODEL_PATH = os.path.join(BASE_DIR, 'predictor', 'model.pkl')
with open(MODEL_PATH, 'wb') as file:
    pickle.dump(model, file)

print("✅ Model trained and saved successfully at:", MODEL_PATH)
