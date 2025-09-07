from flask import Flask, render_template, request
import pandas as pd
import pickle
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import smtplib
from email.message import EmailMessage
import os


gbc_pipeline = pickle.load(open("gbc_pipeline.pkl", "rb"))
abc_pipeline = pickle.load(open("abc_pipeline.pkl", "rb"))
rfc_pipeline = pickle.load(open("rfc_pipeline.pkl", "rb"))
ensemble_pipeline = pickle.load(open("ensemble_pipeline.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        gender = request.form['gender']
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        smoking_history = request.form['smoking_history']
        bmi = float(request.form['bmi'])
        hba1c_level = float(request.form['hba1c_level'])
        blood_glucose = int(request.form['blood_glucose'])
        user_email = request.form['email']

        
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'smoking_history': [smoking_history],
            'bmi': [bmi],
            'HbA1c_level': [hba1c_level],
            'blood_glucose_level': [blood_glucose]
        })

        
        gbc_pred = gbc_pipeline.predict(input_data)[0]
        abc_pred = abc_pipeline.predict(input_data)[0]
        rfc_pred = rfc_pipeline.predict(input_data)[0]
        ensemble_pred = ensemble_pipeline.predict(input_data)[0]

        
        pdf_path = generate_pdf_report(input_data, gbc_pred, abc_pred, rfc_pred, ensemble_pred)
        send_pdf_email(user_email, pdf_path)

       
        return render_template('result.html',
                               gbc=gbc_pred,
                               abc=abc_pred,
                               rfc=rfc_pred,
                               ensemble=ensemble_pred)

    except Exception as e:
        return f"<h3 style='color:red'>❌ Error occurred: {str(e)}</h3>"


def generate_pdf_report(data, gbc, abc, rfc, ensemble):
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    folder = "pdf_reports"
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)

    c = canvas.Canvas(file_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    y = 750

    c.drawString(30, y, " Diabetes Prediction Report")
    y -= 30
    c.drawString(30, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y -= 40
    c.drawString(30, y, "🧍 Patient Info:")
    for key, val in data.iloc[0].items():
        y -= 20
        c.drawString(50, y, f"{key}: {val}")

    y -= 40
    c.drawString(30, y, "Model Predictions:")
    c.drawString(50, y - 20, f"Gradient Boosting: {'Diabetes' if gbc else 'No Diabetes'}")
    c.drawString(50, y - 40, f"AdaBoost: {'Diabetes' if abc else 'No Diabetes'}")
    c.drawString(50, y - 60, f"Random Forest: {'Diabetes' if rfc else 'No Diabetes'}")
    y -= 100
    c.drawString(30, y, f"✅ Final Ensemble Prediction: {'Diabetes' if ensemble else 'No Diabetes'}")

    y -= 50
    c.drawString(30, y, "💡 Diabetes Health Tips:")
    tips = [
        "• Eat a balanced, low-sugar diet.",
        "• Check blood sugar regularly.",
        "• Exercise 30 mins/day.",
        "• Reduce stress.",
        "• Stay hydrated."
    ]
    for tip in tips:
        y -= 20
        c.drawString(50, y, tip)

    c.save()
    return file_path


def send_pdf_email(recipient, pdf_path):
    sender_email = "bhushannalawade331@gmail.com"
    sender_pass = "iwglnpcjekwzsrdl"  

    msg = EmailMessage()
    msg['Subject'] = 'Your Diabetes Report'
    msg['From'] = sender_email
    msg['To'] = recipient
    msg.set_content("here is your personal diabetes prediction report.")

    with open(pdf_path, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(pdf_path)

    msg.add_attachment(file_data, maintype='application', subtype='pdf', filename=file_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, sender_pass)
        smtp.send_message(msg)

if __name__ == '__main__':
    app.run(debug=True)
