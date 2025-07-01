import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load and prepare the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['text'] = df['text'].str.lower()

# Split and vectorize
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5)
mean_cv_score = np.mean(cv_scores)
std_cv_score = np.std(cv_scores)

# Streamlit UI
st.title("📩 SMS Spam Detector")
st.write("Enter a text message below to check if it's Spam or Ham:")

user_input = st.text_area("Message Text", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        user_vec = vectorizer.transform([user_input.lower()])
        prediction = model.predict(user_vec)[0]
        st.subheader("Prediction:")
        st.success(f"The message is **{prediction.upper()}**")

st.markdown("---")
st.subheader("📊 Model Evaluation Metrics")

st.write(f"**Test Accuracy:** {accuracy:.4f}")
st.write(f"**Mean CV Score:** {mean_cv_score:.4f}")
st.write(f"**CV Score Std Dev:** {std_cv_score:.4f}")

if accuracy > mean_cv_score and (accuracy - mean_cv_score) > std_cv_score:
    interpretation = "⚠️ Model might be overfitting."
elif accuracy < mean_cv_score and (mean_cv_score - accuracy) > std_cv_score:
    interpretation = "⚠️ Model might be underfitting."
else:
    interpretation = "✅ Model performance is consistent."

st.write(interpretation)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'], ax=ax)
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig)
