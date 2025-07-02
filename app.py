import os
import zipfile
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ========= Kaggle API Config ========= #
os.environ['KAGGLE_USERNAME'] = 'lawrencelaguidao'  # replace with your Kaggle username
os.environ['KAGGLE_KEY'] = '197e1dbb2ae7602eae7784057ec1e55f'  # replace with your Kaggle API key

# Download the dataset using Kaggle API (only runs if file is missing)
if not os.path.exists('spam.csv'):
    os.system('kaggle datasets download -d uciml/sms-spam-collection-dataset')
    with zipfile.ZipFile("sms-spam-collection-dataset.zip", 'r') as zip_ref:
        zip_ref.extractall()

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['text'] = df['text'].str.lower()

# Vectorization and splitting
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5)
mean_cv_score = np.mean(cv_scores)
std_cv_score = np.std(cv_scores)
cm = confusion_matrix(y_test, y_pred)

# ========= Streamlit UI ========= #
st.title("📩 SMS Spam Detector")
st.write("Classify SMS messages as **Spam** or **Ham** using Naive Bayes.")

user_input = st.text_area("Enter a message:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vec = vectorizer.transform([user_input.lower()])
        prediction = model.predict(input_vec)[0]
        st.success(f"Prediction: **{prediction.upper()}**")

st.markdown("---")
st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** {accuracy:.4f}")
st.write(f"**Mean CV Score:** {mean_cv_score:.4f}")
st.write(f"**CV Std Dev:** {std_cv_score:.4f}")

if accuracy > mean_cv_score and (accuracy - mean_cv_score) > std_cv_score:
    st.warning("⚠️ Model may be overfitting slightly.")
elif accuracy < mean_cv_score and (mean_cv_score - accuracy) > std_cv_score:
    st.warning("⚠️ Model may be underfitting.")
else:
    st.info("✅ Model is performing consistently.")

st.subheader("📉 Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'], ax=ax)
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig)
