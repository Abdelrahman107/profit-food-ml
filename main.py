import json
import pickle
from recommender import Recommender
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
import requests
model = pickle.load(open('model.pkl','rb'))
dataset = pd.read_csv('dataset.csv')
app = Flask(__name__)
url = 'http://127.0.0.1:5000'

# routes


def preprocess_inputs(inputs):
    ob = Recommender()
    data = ob.get_features()
    total_features = data.columns
    d = dict()
    for i in total_features:
        d[i]= 0
    for i in inputs:
        
        d[i] = 1

    final_input = list(d.values())   
    return final_input

def get_recommendation(inputs):
    
    distances , indices = model.kneighbors(inputs)
    df_results = pd.DataFrame(columns=list(dataset.columns))
    for i in list(indices):
        df_results = df_results.append(dataset.loc[i])
    df_results = df_results.filter(['Name','Nutrient','Veg_Non','Price','Review','Diet','Disease','description'])
    df_results = df_results.drop_duplicates(subset=['Name'])
    df_results = df_results.reset_index(drop=True)
    return df_results['Name']



   
@app.route('/', methods=['POST'])
def recommend():

    if request.method == 'POST':
        sample_input = request.get_json(force=True)
        input = request_to_array(sample_input)
        input2= preprocess_inputs(input)
        # try catch get recommendation
        try:

            results = get_recommendation([input2])
            return jsonify(results.to_dict())
        except:
            return jsonify({"error": "Wrong Input to the model or format"}) 



def request_to_array(input):
    #input = json.loads(input)
    arr = []
    for key,values in dict(input).items():
        if type(values) == list:
            for j in values:
                arr.append(j)
        elif type(values) == str:
            y=values.strip('][').split(',')
            for j in y:    
                if(j==""):
                    continue
                arr.append(j.strip('\''))
    return arr        

        
       

if __name__ == '__main__':
    app.run()

    




































   
