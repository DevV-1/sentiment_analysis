# app.py
from flask import Flask, request, render_template, jsonify
import pickle

# Load the trained model
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        prediction = model.predict(data)
        sentiment = ''
        if prediction == 0:
            sentiment = 'Negative'
        elif prediction == 1:
            sentiment = 'Positive'
        else:
            sentiment = 'Neutral'
        
        return render_template('index.html', prediction_text=f'Sentiment: {sentiment}', message=message)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([data['message']])
    sentiment = ''
    if prediction == 0:
        sentiment = 'Negative'
    elif prediction == 1:
        sentiment = 'Positive'
    else:
        sentiment = 'Neutral'
    return jsonify(sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
