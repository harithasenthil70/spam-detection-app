import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

data = [
    "win money now",
    "hello how are you",
    "free prize offer",
    "let's meet tomorrow",
    "claim your reward",
    "good morning"
]

labels = [1,0,1,0,1,0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)

model = LogisticRegression()
model.fit(X, labels)

st.title("Spam Detection App")

msg = st.text_input("Enter message:")

if st.button("Predict"):
    msg_vec = vectorizer.transform([msg])
    result = model.predict(msg_vec)

    if result[0] == 1:
        st.error(" Spam Message")
    else:
        st.success(" Not Spam")