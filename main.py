from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("churn_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({'Churn': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
    
import joblib

joblib.dump(best_model, "churn_model.pkl")
