from flask import Flask, jsonify
from predictions import driver 


app = Flask(__name__)

@app.route('/predict/<comp>')
def predict(comp):
    percentage = driver(comp)
    if percentage == 0:
        return 400
    predict_data = {}
    predict_data["percentage"] = percentage
    if percentage > 0:
        predict_data["winning_team"] = 1
    else:
        predict_data["winning_team"] = 2
    
    return jsonify(predict_data), 200

if __name__  == "__main__":
    app.run(debug=True)