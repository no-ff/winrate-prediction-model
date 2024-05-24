from flask import Flask
from predictions import v2driver 

app = Flask(__name__)

@app.route('/')
def greet():
    return 'Welcome to the noff API'

@app.route('/<comp>')
def predict(comp):
    try:
        prediction_result = v2driver(comp)
        return prediction_result
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__  == "__main__":
    app.run(debug=True)

