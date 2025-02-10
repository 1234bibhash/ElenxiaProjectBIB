# Personalized Health Recommendation System

This project is a **Personalized Health Recommendation System** that utilizes data from wearable devices to assess users' health status and provide customized recommendations. The system is built using **Python, Pandas, NumPy, Scikit-learn, and RandomForestClassifier** for classification.

## Features
- Reads wearable device data from a CSV file.
- Preprocesses the data by standardizing numerical values and categorizing health status.
- Trains a **Random Forest Classifier** to predict user health status.
- Provides personalized health recommendations based on predicted health status.
- Evaluates the model using accuracy and classification reports.

## Installation
### Prerequisites
Ensure you have Python installed along with the required dependencies.

### Install Dependencies
```bash
pip install pandas numpy scikit-learn
```

## Usage
1. **Prepare your dataset**: The CSV file should contain the following columns:
   - `user_id`
   - `steps`
   - `calories_burned`
   - `heart_rate`
   - `sleep_hours`
   - `stress_level`

2. **Run the script**:
```bash
python health_recommendation.py
```

3. **Output**:
   - Displays model accuracy and classification report.
   - Generates **personalized health recommendations** for each user in the dataset.

## Code Overview
- `read_csv_data(file_path)`: Reads data from CSV.
- `preprocess_data(df)`: Processes and scales the data.
- `train_model(X_train, y_train)`: Trains a RandomForestClassifier.
- `generate_recommendations(health_status, user_data)`: Provides customized health advice.
- `get_health_recommendations(user_data, model, scaler)`: Predicts health status and returns recommendations.

## Example Output
```
Model Accuracy: 92%

User ID: 101
Health Status: Good
Personalized Recommendations:
- Try to increase your daily step count to 8500 steps.
- Aim for 7.5 hours of sleep each night.
- Incorporate stress-reduction techniques like meditation.
```

## Future Enhancements
- Support for real-time data streaming from wearable devices.
- Integration with mobile applications for interactive user experience.
- Additional health metrics such as hydration levels and activity intensity.

## License
This project is open-source and available for modification and improvement.

