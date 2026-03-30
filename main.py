# Student Performance Predictor

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("dataset.csv")

# Features and target
X = data[['study_hours', 'attendance', 'sleep_hours', 'previous_marks']]
y = data['final_score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("Model trained successfully!")
print("Mean Absolute Error:", round(mae, 2))

# User input
print("\n--- Student Performance Predictor ---")
study_hours = float(input("Enter study hours per day: "))
attendance = float(input("Enter attendance percentage: "))
sleep_hours = float(input("Enter average sleep hours: "))
previous_marks = float(input("Enter previous exam marks: "))

# Prediction
result = model.predict([[study_hours, attendance, sleep_hours, previous_marks]])

print("\nPredicted Final Score:", round(result[0], 2))
