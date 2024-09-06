import streamlit as st
import pickle

# let us load the vectorizer and naive model
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# transform_test function for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download("stopwords")
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # removing spcl characters and retaining alphanumeric words
    text = [word for word in text if word.isalnum()]
    # removing stopwords
    text = [
        word
        for word in text
        if word not in stopwords.words("english") and word not in string.punctuation
    ]
    # applying stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)


# saving streamlit code

st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter message: ")

if st.button("Predict"):
    # preprocess
    transformed_sms = transform_text(input_sms)
    # vectorize
    vector_input = tfidf.transform([transformed_sms])
    # predict
    result = model.predict(vector_input)[0]
    # display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
