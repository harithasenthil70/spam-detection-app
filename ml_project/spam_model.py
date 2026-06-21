import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = {
    "message": [
        "win money now",
        "hello how are you",
        "claim your prize",
        "let's meet tomorrow",
        "free lottery offer",
        "good morning"
    ],
    "label": [1,0,1,0,1,0]
}

df = pd.DataFrame(data)

X = df["message"]
Y = df["label"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, Y_train)

msg = ["win free money"]
msg_vec = vectorizer.transform(msg)

print(model.predict(msg_vec))