from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import seaborn as sb
import plotly
import plotly.graph_objs as go
# Data dari flask di kirim ke browser dalam bentuk json
import json
import joblib
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

app = Flask(__name__)

# Sumber data
SBA = pd.read_csv('datafix.csv')

@app.route('/')
def home():
    return render_template('index.html')

# Render Picture
@app.route('/static/<path:x>')
def gal(x):
    return send_from_directory("static",x)

# Render About page
@app.route('/about')
def about():
    return render_template('about.html')

# Prediction Page
@app.route('/predict')
def predict():
    return render_template('predict.html')

# Result Page
import re,string
import pandas as pd
from stopwords_id import stop_words
string.punctuation

# membaca file normalisasi
df_norm = pd.read_csv("normal.txt")
# membuat kamus normalisasi (dictionary)
df_kamus = {}
for dt in df_norm.itertuples():
    df_kamus[dt[1]] = dt[2]

 # kata-kata yang harus dihapus
    word_to_remove = ['username','url']

def preprocess(row):
  # case folding
  row = row.lower()

  # hapus menghapus
  row = re.sub(r"(?:\@|#|\d)\S+", "", row)
    
  # hapus numerik
  row = re.sub(r"\d+", "", row)

  # normalisasi kata
  row = ' '.join([df_kamus[a] if a in df_kamus else a for a in row.split()])

  # hapus kata2 tertentu
  row = ' '.join([a for a in row.split() if a not in word_to_remove])

  # ganti tanda baca jadi spasi
  row = row.translate(str.maketrans(string.punctuation, " "*len(string.punctuation)))
    
  # normalisasi kata
  row = ' '.join([df_kamus[a] if a in df_kamus else a for a in row.split()])

  # hapus stop words
  row = ' '.join([a for a in row.split() if a not in stop_words()])
  return row

@app.route('/SBA_Loan_Result', methods=["POST", "GET"])
def SBA_Loan_predict():
    if request.method == "POST":
        input = request.form
        twitter = preprocess(input['twitter'])

# Term, NewExist, GrAppv, SBA_Appv, RevLineCr, Lowdoc, NAICS_11
        pred = klasifikasi.predict([twitter])
        proba = klasifikasi.predict_proba([twitter])
        pred_and_proba = f"{round(np.max(proba)*100,2)}% {pred[0]}"


        return render_template('result.html',
        data=input, prediction=pred_and_proba, komen=input['twitter'], twitter=(preprocess(input['twitter'])))

if __name__ == '__main__':
    
    
    klasifikasi = joblib.load('putripickle.pkl')
    app.run(debug=True, port=4000)