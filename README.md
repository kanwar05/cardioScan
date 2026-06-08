# CardioScan

CardioScan is a Streamlit web app for predicting cardiovascular disease risk from common clinical risk factors. It uses a trained K-Nearest Neighbors classifier in the live app and includes a notebook for data exploration, preprocessing, and model comparison.

## What this project includes

- `app.py`: Streamlit UI for user input, preprocessing, and model inference.
- `heart.csv`: raw dataset used for exploration and model training.
- `heart.ipynb`: Jupyter notebook for EDA, preprocessing, model training, and accuracy comparison.
- `KNN_heart_model.pkl`: saved KNN model used by the app.
- `heart_scaler.pkl`: saved standard scaler fitted on the training features.
- `heart_columns.pkl`: saved expected feature column order for model input.
- `README.md`: project overview and instructions.

## Project workflow

1. Load the `heart.csv` dataset.
2. Clean missing or invalid values by replacing zeros in `Cholesterol` and `RestingBP` with the feature mean.
3. One-hot encode categorical variables with `pd.get_dummies(drop_first=True)`.
4. Split the data into training and test sets using `train_test_split(test_size=0.20, random_state=42)`.
5. Apply `StandardScaler` to the numeric features.
6. Train and evaluate several models.
7. Save the best model artifacts with `joblib` for the Streamlit app.

## Models compared

The notebook trains and evaluates these classifiers using the same preprocessed training and test split:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)

### Accuracy comparison table

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Logistic Regression | 0.8696   | 0.8846   |
| KNN                 | 0.8641   | 0.8815   |
| Gaussian NB         | 0.8478   | 0.8614   |
| Decision Tree       | 0.7989   | 0.8159   |
| SVM                 | 0.8478   | 0.8667   |

The app uses the KNN model because it performed well in this comparison and is a simple classifier for the deployed inference flow.

## Features

- Interactive Streamlit interface with clean layout.
- Dark/light theme support using custom CSS.
- Inputs for core clinical variables, including:
  - Age
  - Sex
  - Resting blood pressure
  - Cholesterol
  - Fasting blood sugar
  - Resting ECG results
  - Chest pain type
  - Exercise-induced angina
  - ST slope
  - Maximum heart rate
  - Oldpeak
- Uses preprocessing artifacts from training time so live input matches the model training pipeline.
- Displays a clear risk prediction result with a medical disclaimer.

## How to run

Install the dependencies:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

Then run the app from the project folder:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Notes on files

- `app.py` loads the saved KNN model, scaler, and feature columns with `joblib`.
- `heart.ipynb` contains the exploratory data analysis, preprocessing, and model performance comparison.
- `heart.csv` is the dataset used to build and validate the models.

## Disclaimer

CardioScan is for educational and informational purposes only. It is not a medical device and does not replace professional medical advice, diagnosis, or treatment.

## Author

Built by Kanwar.
