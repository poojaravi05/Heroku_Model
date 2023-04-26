# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
dataset = pd.read_csv('hiring.csv')

# Fill missing values with appropriate values
dataset['experience'].fillna('zero', inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

# Convert words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

dataset['experience'] = dataset['experience'].apply(lambda x : convert_to_int(x))

# Prepare the feature matrix and target vector
X = dataset.iloc[:, :3]
y = dataset.iloc[:, -1]

# Assign column names to X
X.columns = ['experience', 'test_score', 'interview_score']

# Create and fit the model
regressor = LinearRegression()
regressor.fit(X, y)

# Save the model to disk
pickle.dump(regressor, open('model.pkl','wb'))
