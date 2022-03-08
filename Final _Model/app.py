
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

df_1 = pd.read_csv("atr_emp.csv")

q = ""

@app.route("/")
def loadPage():
    return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def predict():
    '''
    Age
    Gender
    Annual_Salary
    CurrentPayGrade
    TimewithCompany(yrs)
    DistancetoWork(miles)
    Dept_Name
    '''

    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']

    model = pickle.load(open("model.sav", "rb"))

    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7
             ]]

    new_df = pd.DataFrame(data, columns=['Age', 'Gender', 'Annual_Salary', 'CurrentPayGrade',
                          'TimewithCompany(yrs)', 'DistancetoWork(miles)', 'Dept_Name'])

    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    new_df__dummies = pd.get_dummies(df_2[['Age', 'Gender', 'Annual_Salary', 'CurrentPayGrade',
                                     'TimewithCompany(yrs)', 'DistancetoWork(miles)', 'Dept_Name']])


    single = model.predict(new_df__dummies.tail(1))
    probablity = model.predict_proba(new_df__dummies.tail(1))[:, 1]


    if single == 1:
        o1 = "This employee is likely to continue working here!!"
        o2 = "Confidence: {}".format(probablity*100)
    else:
        o1 = "This employee is likely to be leave!!"
        o2 = "Confidence: {}".format(probablity*100)

    return render_template('home.html', output1=o1, output2=o2,
                           query1=request.form['query1'],
                           query2=request.form['query2'],
                           query3=request.form['query3'],
                           query4=request.form['query4'],
                           query5=request.form['query5'],
                           query6=request.form['query6'],
                           query7=request.form['query7']
                           )


if __name__ == "__main__":
    app.run()
