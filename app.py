import streamlit as st
import pickle

# let us load the vectorizer and naive model
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# saving streamlit code

st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter message: ")

if st.button("Predict"):
    # preprocess
    transformed_sms = transformed_test(input_sms)
    # vectorize
    vector_iput = tfidf.transform([transform_sms])
    # predict
    result = model.predict(vector_input)[0]
    # display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")
