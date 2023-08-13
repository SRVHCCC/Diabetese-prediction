import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

classifire=pickle.load(open('/config/workspace/models/classifire.pkl','rb'))
scaler=pickle.load(open('/config/workspace/models/lscaler.pkl','rb'))


@app.route("/")
def hello_world():
    return "hello world"

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Pregnancies=int(request.form.get('Pregnancies'))
        Glucose=int(request.form.get('Glucose'))
        BloodPressure=int(request.form.get('BloodPressure'))
        SkinThickness=float(request.form.get('SkinThickness'))
        Insulin=float(request.form.get('Insulin'))
        BMI=float(request.form.get('BMI'))
        DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction'))
        Age=int(request.form.get('Age'))

        new_scaled_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        result=classifire.predict(new_scaled_data)

        if result[0]==1:
            results="Diabetic"

        else:
            results="NO Diabetic"

        

        return render_template('home.html',result=results)




        
    else:
        return render_template('home.html')





if __name__=="__main__":
    app.run(host="0.0.0.0")
