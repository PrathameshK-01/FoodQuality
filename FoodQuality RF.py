import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
'''
# Generate and save the dataset
n_samples = 1000

data = {
    'Color': np.random.randint(0, 256, n_samples),  # Color intensity in arbitrary units
    'Smell': np.random.choice(['Good', 'Moderate', 'Poor'], n_samples),  # Categorical
    'Taste': np.random.choice(['Good', 'Moderate', 'Poor'], n_samples),  # Categorical
    'Texture': np.random.choice(['Good', 'Moderate', 'Poor'], n_samples),  # Categorical
    'pH_Level': np.random.uniform(3, 9, n_samples),  # pH level
    'Moisture_Content': np.random.uniform(0, 100, n_samples),  # Moisture content percentage
    'Nutrient_Content': np.random.uniform(0, 100, n_samples),  # Nutrient content percentage
    'Temperature': np.random.uniform(-10, 50, n_samples),  # Temperature in Celsius
}

def generate_quality_result(row):
    if row['Color'] > 200 or row['Smell'] == 'Poor' or row['Taste'] == 'Poor' or row['Texture'] == 'Poor':
        return 'Poor'
    elif row['Color'] > 150 or row['Smell'] == 'Moderate' or row['Taste'] == 'Moderate' or row['Texture'] == 'Moderate':
        return 'Moderate'
    else:
        return 'Good'

df = pd.DataFrame(data)
df['QualityResult'] = df.apply(generate_quality_result, axis=1)
df.to_csv('food_quality_dataset.csv', index=False)

print("Dataset generated and saved to food_quality_dataset.csv")
'''
# Load the dataset and prepare it for training
df = pd.read_csv('food_quality_dataset.csv')

label_encoders = {}
for column in ['Smell', 'Taste', 'Texture']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

le_quality = LabelEncoder()
df['QualityResult'] = le_quality.fit_transform(df['QualityResult'])

X = df.drop('QualityResult', axis=1)
y = df['QualityResult']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Function to get user input and predict food quality
def get_user_input():
    color = int(input("Enter Color intensity (0-255): "))
    smell = input("Enter Smell (Good/Moderate/Poor): ")
    taste = input("Enter Taste (Good/Moderate/Poor): ")
    texture = input("Enter Texture (Good/Moderate/Poor): ")
    ph_level = float(input("Enter pH Level: "))
    moisture_content = float(input("Enter Moisture Content (%): "))
    nutrient_content = float(input("Enter Nutrient Content (%): "))
    temperature = float(input("Enter Temperature (Celsius): "))

    smell_encoded = label_encoders['Smell'].transform([smell])[0]
    taste_encoded = label_encoders['Taste'].transform([taste])[0]
    texture_encoded = label_encoders['Texture'].transform([texture])[0]

    user_data = pd.DataFrame({
        'Color': [color],
        'Smell': [smell_encoded],
        'Taste': [taste_encoded],
        'Texture': [texture_encoded],
        'pH_Level': [ph_level],
        'Moisture_Content': [moisture_content],
        'Nutrient_Content': [nutrient_content],
        'Temperature': [temperature]
    })

    return user_data

# Get real-time input from the user
user_input = get_user_input()

# Predict the quality result
user_pred = clf.predict(user_input)
user_quality = le_quality.inverse_transform(user_pred)

print(f'Predicted Food Quality: {user_quality[0]}')
