from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import accuracy_score
import googletrans
from googletrans import Translator
from google_trans_new import google_translator
from tqdm.auto import tqdm
import streamlit as st
import pandas as pd
import numpy as np
import regex as re
import json
import nltk
import pickle
nltk.download('stopwords')
nltk.download('punkt')

# user interface
st.title("""Aplikasi Analisis Sentimen PSE""")
st.subheader('Input Teks')
new_data = st.text_area('Masukkan kalimat yang akan dianalisis :')
submit = st.button("submit")

if submit:
    #preprocessing
    def preprocessing(word):
        lower_case = new_data.lower()
        clean_tweet = re.sub("@[A-Za-z0-9_]+", "",lower_case)  #clenasing mention
        clean_tweet1 = re.sub("#[A-Za-z0-9_]+", "",clean_tweet)  #clenasing hashtag
        clean_tweet2 = re.sub(r'http\S+', '', clean_tweet1) #cleansing url link
        clean_tweet3 = re.sub("[^a-zA-Z ]+", " ", clean_tweet2) # cleansing character
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stem = stemmer.stem(clean_tweet3)
        tokens = word_tokenize(stem)
        words = []
        temp = []
        

    # Dataset
    df = pd.read_csv(
        'https://raw.githubusercontent.com/Niken-Amalia/Informatika_Pariwisata/main/hasillabelling%20-%20Sheet1.csv')

    names = []
    with open(r'C:\Users\HP 14\hh.txt', 'r') as fp:
        for line in fp:
            x = line[:-1]
            names.append(x)

    #save model dengan pickle
    with open('model.pkl', 'rb') as file:
        model_pkl = pickle.load(file)

    #ekstraksi fitur
    tfidfvectorizer = TfidfVectorizer(analyzer='word')
    tfidf_wm = tfidfvectorizer.fit_transform(names)
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
 
     #split data menjadi training data(80%) dan testing data(20%)
    x_train, x_test = train_test_split(tfidf_wm, test_size=0.2, random_state=1)
    training_label, test_label = train_test_split(
        df['Sentiment'], test_size=0.2, random_state=1)  # Nilai Y training dan Nilai Y testing

    #modelling dengan logistic regression 
    model_LR = model_pkl.fit(x_train, training_label)

    y_pred = model_LR.predict(test)

    # Prediksi
    tfidf_inputan = tfidfvectorizer.transform([stem]).toarray()  # vectorizing
    pred_text = model_pkl.predict(tfidf_inputan)
    st.subheader('Prediksi Kelas Teks')
    st.info(pred_text)

    # akurasi
    y_pred = model_pkl.predict(test)
    akurasi = accuracy_score(test_label, y_pred)
    st.subheader('Akurasi Model')
    st.success(akurasi)