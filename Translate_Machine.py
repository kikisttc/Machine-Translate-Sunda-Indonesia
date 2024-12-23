import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC  # Menggunakan LinearSVC untuk klasifikasi teks
from sklearn.pipeline import Pipeline
import pickle
import streamlit as st


# 2. Load dataset
file_path = './Kamus_Bahasa_Sunda_Indonesia.csv'
data = pd.read_csv(file_path)

print(data.head())

# 3. Preprocessing
data['Sunda'] = data['Sunda'].astype(str).fillna('')
data['Indonesia'] = data['Indonesia'].astype(str).fillna('')

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['Sunda'], data['Indonesia'], test_size=0.4, random_state=42
)

# 5. Create pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', LinearSVC(
    )),  # Menggunakan LinearSVC sebagai classifier
])

# 6. Train model
model.fit(X_train, y_train)

# 7. Function to translate and handle unknown words
def translate_sunda_to_indonesia(text):
    try:
        predicted_translation = model.predict([text])[0]
    except KeyError:  # Handle KeyError if word not in vocabulary
        predicted_translation = "Unknown"  # Or handle differently, e.g., return original word
    return predicted_translation

# (Optional) Evaluate the model
# 8. Test the translation
sunda_text = "anggo"  # Replace with the Sunda text you want to translate
indonesia_translation = translate_sunda_to_indonesia(sunda_text)
print(f"Sunda: {sunda_text}")
print(f"Indonesia: {indonesia_translation}")

# (Optional) Evaluate the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy}")


model_path = "./TMM_Sunda_Indonesia_model.pkl"

with open(model_path, "wb") as file:  # Use 'wb' for writing binary
    pickle.dump(model, file)


# Muat model
with open(model_path, "rb") as file:
    model = pickle.load(file)

    st.title("Penerjemah Sunda-Indonesia")


# Input teks Sunda
sunda_text = st.text_input("Masukkan kata Sunda:", "")

# Tombol terjemahkan
if st.button("Terjemahkan"):
    # Prediksi terjemahan
    indonesia_translation = model.predict([sunda_text])[0]

    # Tampilkan hasil
    st.write("Terjemahan Bahasa Indonesia:", indonesia_translation)