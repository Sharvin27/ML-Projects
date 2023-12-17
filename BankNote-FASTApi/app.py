#Library imports

import uvicorn #For ASGI
from fastapi import FastAPI
from banknotes import BankNote
import numpy as numpy
import pickle
import pandas as pandas

app = FastAPI()

load_model = pickle.load(open('classifier.pkl','rb'))

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return{'message' : 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome' : f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence

@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.model_dump()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']

    prediction = load_model.predict([[variance, skewness, curtosis, entropy]])

    if(prediction[0] > 0.5):
        prediction = "Fake Note"
    else:
        prediction = "it is a bank note"
    return{
        'prediction': prediction
    }



if __name__ == '__main__':
    uvicorn.run(app,host = '127.0.0.1', port =80000)

    #TO RUN python -m uvicorn app:app --reload