import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle

# Chargement des données
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header=None)
df.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)
df["target"] = df["target"].apply(lambda x: 1 if int(x) > 0 else 0)

# Séparation des données
X = df.drop(["target"], axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraînement du modèle (choix : Random Forest)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sauvegarde du modèle et du scaler
with open("best_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# Chargement du modèle et du scaler
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Interface Streamlit
st.title("🩺 Prédiction des Maladies Cardiaques")

st.write("Remplissez les informations du patient pour estimer son risque de maladie cardiaque.")

# Formulaire de saisie des caractéristiques
age = st.slider("Âge", 20, 80, 50)
sex = st.radio("Sexe", [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
cp = st.selectbox("Type de douleur thoracique (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Pression artérielle au repos (trestbps)", 80, 200, 120)
chol = st.number_input("Cholestérol sérique (chol)", 100, 600, 200)
fbs = st.radio("Glycémie à jeun > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Résultats électrocardiographiques au repos (restecg)", [0, 1, 2])
thalach = st.number_input("Fréquence cardiaque maximale atteinte (thalach)", 70, 220, 150)
exang = st.radio("Angine induite par l'exercice (exang)", [0, 1])
oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 6.0, 1.0)
slope = st.selectbox("Pente du segment ST (slope)", [0, 1, 2])
ca = st.selectbox("Nombre de vaisseaux majeurs colorés par fluoroscopie (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassémie (thal)", [3, 6, 7])

# Préparer les données pour la prédiction
user_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
user_data_scaled = scaler.transform(user_data)

# Bouton de prédiction
if st.button("Prédire"):
    prediction = model.predict(user_data_scaled)
    prob = model.predict_proba(user_data_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Le modèle prédit une maladie cardiaque avec une probabilité de {prob:.2f}.")
    else:
        st.success(f"✅ Le modèle ne détecte pas de maladie cardiaque. Probabilité: {prob:.2f}")
