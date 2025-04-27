from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# URL of your FastAPI backend
FASTAPI_URL = "http://127.0.0.1:8000/recommend"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    user_input = request.form['user_input']
    
    # Prepare the request payload
    payload = {"text": user_input}
    
    # Send the POST request to FastAPI backend
    response = requests.post(FASTAPI_URL, json=payload)
    
    if response.status_code == 200:
        recommendations = response.json()['results']
        return render_template('index.html', recommendations=recommendations, user_input=user_input)
    else:
        return render_template('index.html', error="Error fetching recommendations.")

if __name__ == "__main__":
    app.run(debug=True)
