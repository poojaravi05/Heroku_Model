import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model from disk
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Get input values from form
    try:
        int_features = [int(x) for x in request.form.values()]
    except ValueError:
        return render_template('index.html', error_msg='Invalid input! Please enter numerical values only.')
    
    if len(int_features) != 3:
        return render_template('index.html', error_msg='Invalid input! Please enter all 3 values.')
    
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Predicted Salary shall be $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
