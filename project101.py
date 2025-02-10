import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def read_csv_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Create health_status column based on the other features
    df['health_status'] = np.where(
        (df['steps'] > 10000) & (df['calories_burned'] > 2500) & 
        (df['heart_rate'] < 100) & (df['sleep_hours'] > 7) & (df['stress_level'] < 3),
        'Excellent',
        np.where(
            (df['steps'] > 7000) & (df['calories_burned'] > 2000) & 
            (df['heart_rate'] < 110) & (df['sleep_hours'] > 6) & (df['stress_level'] < 4),
            'Good',
            'Needs Improvement'
        )
    )
    
    X = df.drop(['user_id', 'health_status'], axis=1)
    y = df['health_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def generate_recommendations(health_status, user_data):
    recommendations = {
        'Excellent': [
            "Maintain your current routine, you're doing great!",
            "Consider adding variety to your workouts to keep challenging yourself.",
            "Share your success with friends and family to inspire them."
        ],
        'Good': [
            f"Try to increase your daily step count to {user_data['steps'] + 1000} steps.",
            f"Aim for {user_data['sleep_hours'] + 0.5} hours of sleep each night.",
            "Incorporate stress-reduction techniques like meditation or deep breathing exercises."
        ],
        'Needs Improvement': [
            f"Gradually increase your daily step count. Aim for {user_data['steps'] + 2000} steps per day.",
            "Set a regular sleep schedule and aim for 7-9 hours of sleep per night.",
            f"Try to burn {user_data['calories_burned'] + 300} calories through physical activity.",
            "Practice stress management techniques like mindfulness or yoga.",
            "Consider consulting with a healthcare professional for personalized advice."
        ]
    }
    
    return recommendations[health_status]

def get_health_recommendations(user_data, model, scaler):
    user_data_scaled = scaler.transform(user_data.values.reshape(1, -1))
    health_status = model.predict(user_data_scaled)[0]
    recommendations = generate_recommendations(health_status, user_data)
    
    return health_status, recommendations

if __name__ == "__main__":
    # Read CSV data
    csv_file_path = 'wearable_data.csv'  # Replace with your CSV file path
    df = read_csv_data(csv_file_path)
    
    # Preprocess the data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)
    
    # Train the model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Get recommendations for each user in the CSV
    for _, user_data in df.iterrows():
        health_status, recommendations = get_health_recommendations(user_data.drop(['user_id', 'health_status']), model, scaler)
        
        print(f"\nUser ID: {user_data['user_id']}")
        print(f"Health Status: {health_status}")
        print("Personalized Recommendations:")
        for rec in recommendations:
            print("-", rec)