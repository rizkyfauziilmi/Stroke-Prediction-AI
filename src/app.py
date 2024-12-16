import gradio as gr
import joblib
import numpy as np

# Load model
model = joblib.load('../bin/model.pkl')

def predict_stroke(gender: str, age: int, hypertension: str, heart_disease: str, ever_married: str, work_type: str, residence_type: str, avg_glucose_level: float, weight: float, height: float, smoking_status: str) -> tuple:
    # Kalkulasi BMI
    bmi = weight / (height / 100) ** 2

    # Mapping data
    gender_map = {"Male": 1, "Female": 0, "Other": 2}
    yes_no_map = {"Yes": 1, "No": 0}
    work_type_map = {"Private": 2, "Self Employed": 3, "Government Job": 0, "Children": 4, "Never Worked": 1}
    residence_type_map = {"Urban": 1, "Rural": 0}
    smoking_status_map = {"formerly smoked": 1, "never smoked": 2, "smokes": 3, "Unknown": 0}

    gender = gender_map[gender]
    hypertension = yes_no_map[hypertension]
    heart_disease = yes_no_map[heart_disease]
    ever_married = yes_no_map[ever_married]
    work_type = work_type_map[work_type]
    residence_type = residence_type_map[residence_type]
    smoking_status = smoking_status_map[smoking_status]

    # input untuk melakukan prediksi
    input_data = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]])

    # Prediksi dan probability
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Probability of stroke

    # Jika hasil prediksi 1, maka pasien memiliki risiko stroke tinggi dan sebaliknya
    if prediction[0] == 1:
        stroke_risk = f"High risk of stroke (Probability: {probability * 100:.2f}%)"
        explanation = "Factors contributing to high risk: "
        if age > 60:
            explanation += "Age over 60, "
        if hypertension == 1:
            explanation += "Hypertension, "
        if heart_disease == 1:
            explanation += "Heart Disease, "
        if bmi > 25:
            explanation += "High BMI, "
        if smoking_status in [1, 3]:
            explanation += "Smoking, "
        explanation = explanation.rstrip(", ") + "."
    else:
        stroke_risk = f"Low risk of stroke (Probability: {probability * 100:.2f}%)"
        explanation = "Factors contributing to low risk: "
        if age <= 60:
            explanation += "Age 60 or below, "
        if hypertension == 0:
            explanation += "No Hypertension, "
        if heart_disease == 0:
            explanation += "No Heart Disease, "
        if bmi <= 25:
            explanation += "Normal BMI, "
        if smoking_status == 2:
            explanation += "Non-Smoker, "
        explanation = explanation.rstrip(", ") + "."
        
    bmi_description = f"BMI: {bmi:.2f} (Weight: {weight} kg, Height: {height} cm)"
        
    return stroke_risk, bmi_description, explanation

# definisi gradio interface
inputs = [
    gr.Radio(["Male", "Female", "Other"], label="Gender"),
    gr.Number(label="Age", value=25, precision=0),
    gr.Radio(["Yes", "No"], label="Hypertension"),
    gr.Radio(["Yes", "No"], label="Heart Disease"),
    gr.Radio(["Yes", "No"], label="Ever Married"),
    gr.Dropdown(["Private", "Self Employed", "Government Job", "Children", "Never Worked"], label="Work Type"),
    gr.Radio(["Urban", "Rural"], label="Residence Type"),
    gr.Number(label="Average Glucose Level", value=100.0),
    gr.Number(label="Weight (kg)", value=70.0),
    gr.Number(label="Height (cm)", value=170.0),
    gr.Dropdown(["formerly smoked", "never smoked", "smokes", "Unknown"], label="Smoking Status")
]

outputs = [
    gr.Textbox(label="Stroke Prediction"),
    gr.Textbox(label="BMI Calculation"),
    gr.Textbox(label="Prediction Explanation")
]

app = gr.Interface(
    fn=predict_stroke,
    inputs=inputs,
    outputs=outputs,
    title="Stroke Risk Predictor",
    description="Input patient data to predict stroke risk",
    flagging_mode='auto'
)

app.launch()