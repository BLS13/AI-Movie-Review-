from flask import Flask, request, render_template
import joblib
import re
import numpy as np

try:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    MODEL_CLASSES = list(model.classes_)

except FileNotFoundError:
    print("Error: 'model.pkl' or 'vectorizer.pkl' not found. Ensure they are in the same directory.")
    exit()

app = Flask(__name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    probability = None
    review_text = ""

    if request.method == 'POST':
        review_text = request.form['review_text']
        
        if not review_text.strip():
            return render_template('ui.html', sentiment=sentiment, probability=probability, review_text=review_text)

        try:
            cleaned_text = clean_text(review_text)
            X_new = vectorizer.transform([cleaned_text])
            
            prediction = model.predict(X_new)[0]
            proba = model.predict_proba(X_new)[0]
            
            predicted_class_index = MODEL_CLASSES.index(prediction)
            
            confidence = proba[predicted_class_index]
            
            sentiment = prediction.capitalize() 
            probability = f"{confidence * 100:.2f}"
            
        except ValueError as e:
            print(f"Error during analysis: {e}")
            sentiment = "Error"
            probability = "N/A"
            
    return render_template(
        'ui.html',  
        sentiment=sentiment, 
        probability=probability, 
        review_text=review_text
    )

if __name__ == '__main__':
    app.run(debug=True)

