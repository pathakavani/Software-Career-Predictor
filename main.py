from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn import preprocessing
import joblib

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/form1', methods=['GET', 'POST'])
def form1():
    return render_template('index1.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        print("inside if")
        pos = request.form['pos']
        pa = request.form['pa']
        pse = request.form.get('pse')
        pcn = request.form.get('pcn')
        pcoa = request.form.get('pcoa')
        pm = request.form.get('pm')
        pe = request.form.get('pe')

        hack = request.form.get('hack')
        coding = request.form.get('coding')
        extra = request.form.get('extra')
        wc1 = request.form.get('wc1')
        wc2 = request.form.get('wc2')
        is1 = request.form.get('is1')

        comm = request.form.get('comm')
        speak = request.form.get('speak')
        mem = request.form.get('mem')
        logical = request.form.get('logical')
        rw = request.form.get('rw')
        team = request.form.get('team')
        ie = request.form.get('ie')

        screen = request.form.get('screen')
        job = request.form.get('job')
        comp = request.form.get('comp')
        ica = request.form.get('ica')
        learn = request.form.get('learn')
        hrs = request.form.get('hrs')
        mt = request.form.get('mt')

        dataset = pd.read_csv("roo_data.csv")  #read dataset

        for i in range(len(dataset)):
            if dataset.loc[
                i, "Suggested Job Role"] == 'Network Security Administrator' or \
                    dataset.loc[
                        i, "Suggested Job Role"] == 'Network Engineer' or \
                    dataset.loc[
                        i, "Suggested Job Role"] == 'Network Security Engineer':
                dataset.loc[i, "Suggested Job Role"] = 'Network Admin'

            if dataset.loc[
                i, "Suggested Job Role"] == 'Business Systems Analyst' or \
                    dataset.loc[
                        i, "Suggested Job Role"] == 'Business Intelligence Analyst' or \
                    dataset.loc[
                        i, "Suggested Job Role"] == 'CRM Business Analyst':
                dataset.loc[i, "Suggested Job Role"] = 'Business Analyst'

            if dataset.loc[i, "Suggested Job Role"] == 'Database Manager' or \
                    dataset.loc[
                        i, "Suggested Job Role"] == 'Database Administrator':
                dataset.loc[i, "Suggested Job Role"] = 'Database Developer'

            if dataset.loc[
                i, "Suggested Job Role"] == 'Information Security Analyst' or \
                    dataset.loc[
                        i, "Suggested Job Role"] == 'Information Technology Auditor' or \
                    dataset.loc[
                        i, "Suggested Job Role"] == 'Information Technology Manager':
                dataset.loc[i, "Suggested Job Role"] = 'IT Admin'

            if dataset.loc[
                i, "Suggested Job Role"] == 'Software Systems Engineer' or \
                    dataset.loc[
                        i, "Suggested Job Role"] == 'Software Developer' or \
                    dataset.loc[
                        i, "Suggested Job Role"] == 'Software Quality Assurance (QA) / Testing':
                dataset.loc[i, "Suggested Job Role"] = 'Software Engineer'

            if dataset.loc[
                i, "Suggested Job Role"] == 'CRM Technical Developer' or \
                    dataset.loc[
                        i, "Suggested Job Role"] == 'Technical Support' or \
                    dataset.loc[
                        i, "Suggested Job Role"] == 'Technical Services/Help Desk/Tech Support':
                dataset.loc[i, "Suggested Job Role"] = 'Technical Engineer'

            if dataset.loc[i, "Suggested Job Role"] == 'Solutions Architect':
                dataset.loc[i, "Suggested Job Role"] = 'Data Architect'

            if dataset.loc[i, "Suggested Job Role"] == 'Design & UX':
                dataset.loc[i, "Suggested Job Role"] = 'UX Designer'

            if dataset.loc[i, "Suggested Job Role"] == 'Portal Administrator' or \
                    dataset.loc[
                        i, "Suggested Job Role"] == 'Systems Security Administrator':
                dataset.loc[i, "Suggested Job Role"] = 'Administrator'

            if dataset.loc[
                i, "Suggested Job Role"] == 'Mobile Applications Developer':
                dataset.loc[i, "Suggested Job Role"] = 'Applications Developer'

        data = dataset.iloc[:, :-1].values
        label = dataset.iloc[:, -1].values

        comm = int(comm)
        comm = comm * 10

        input1 = np.array([[int(pos), int(pa), int(pse), int(pcn), int(pe),
                            int(pcoa), int(pm), int(comm), int(hrs),
                            int(logical), int(hack), int(coding), int(speak),
                            screen, learn, extra, wc1, wc2, rw, mem, is1, ica,
                            job, comp, mt, team, ie]])
        data1 = data
        data1 = np.append(data1, input1, axis=0)

        # ---------------conversion of all categorial column values to vector/numerical--------#
        labelencoder = preprocessing.LabelEncoder()
        for i in range(13, 27):
            data1[:, i] = labelencoder.fit_transform(data1[:, i])

        input1 = data1[-1]
        input1 = np.array([input1])

        print(input1)
        # input1 = [[78,89,94,56,78,78,56,67,6,7,3,7,9,'yes','yes','yes','r programming','data science','medium','medium','networks','developer','job','BPA','Management','yes','no']]

        # --------------normalizing the non-categorial column values---------#
        from sklearn.preprocessing import Normalizer
        data2 = input1[:, :13]
        normalized_data1 = Normalizer().fit_transform(data2)

        # In[177]:
        data3 = input1[:, 13:]

        df2 = np.append(normalized_data1, data3, axis=1)

        clf = joblib.load('model1.pkl')

        label1 = labelencoder.fit_transform(label)

        probs = clf.predict_proba(df2)
        best_n = np.argsort(probs, axis=1)
        best_n = best_n[:, -4:-1]


        res = []
        for i in range(0, 3):
            res1 = labelencoder.inverse_transform([best_n[0, i]])
            res.append(res1)

        print(res)

        print("done")

    return render_template('result.html', var1=res[0][0], var2=res[1][0],
                           var3=res[2][0])

@app.route('/analysis')
def analysis():
    return render_template('ana.html')

if __name__ == '__main__':
    app.run(debug=True)