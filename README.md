# CardioScan

CardioScan is a Streamlit web app that predicts cardiovascular disease risk from common clinical inputs. It uses a trained K-Nearest Neighbors model with saved preprocessing artifacts to produce an instant low-risk or elevated-risk assessment.

## Features

- Interactive Streamlit interface
- Dark and light theme toggle
- Inputs for age, biological sex, blood pressure, cholesterol, fasting blood sugar, ECG results, exercise angina, ST slope, maximum heart rate, and oldpeak
- Saved KNN model inference using `joblib`
- Input scaling and column alignment with training-time artifacts
- Clear prediction result with a medical disclaimer

## Project Files

```text
cardioScan/
|-- app.py
|-- KNN_heart_model.pkl
|-- heart_scaler.pkl
|-- heart_columns.pkl
`-- README.md
```

## Requirements

This project needs Python 3.9 or newer.

Install the required Python packages:

```bash
pip install streamlit pandas joblib scikit-learn
```

## How to Run

From the project folder, start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Model Artifacts

The app depends on these files being present in the same folder as `app.py`:

- `KNN_heart_model.pkl`: trained KNN classifier
- `heart_scaler.pkl`: fitted scaler used before prediction
- `heart_columns.pkl`: expected feature column order

Do not rename or move these files unless you also update the paths in `app.py`.

## Usage

1. Enter the patient's clinical information in the form.
2. Click **Run Risk Assessment**.
3. Review the predicted cardiovascular risk result.

## Important Disclaimer

CardioScan is for educational and informational purposes only. It is not a medical device and does not replace diagnosis, treatment, or advice from a qualified healthcare professional.

## Author

Built by Kanwar.
