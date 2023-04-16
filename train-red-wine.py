from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from joblib import dump
import pandas as pd
import neptune

# Initialisierung des Runs, bitte Projekt und API Token einfügen
# https://neptune.ai/
run = neptune.init_run(
    project="davidmelzer/NeptuneDemo-Wine",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vbmV3LXVpLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMDg3NDFlYy0zOTZjLTQ4NjQtYjhlOC02NjI3ZWVmYWVkNjkifQ==",
    source_files=["*.py", "requirements.txt"],
)

# Daten laden und als CSV speichern
data = load_wine()
df = pd.DataFrame(data=data['data'], columns=data['feature_names'])
df.to_csv('red-wine.csv', sep=';', index=False)

# Daten dem Run hinzufügen
run["train_dataset"].track_files("red-wine.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.4, random_state=1234
)

# Parameter für den Random Forest
params = {
    "n_estimators": 10,
    "max_depth": 3,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "max_features": 3,
}

# Parameter dem Run hinzufügen
run["parameters"] = params

# Tags dem Run hinzufügen
run["sys/tags"].add(["david", "red-wine", "random-forest"])

# Modell trainieren und Metriken tracken
clf = RandomForestClassifier(**params)

for epoch in range(100):
    # Modell trainieren
    clf.fit(X_train, y_train)
    # Proba Vektoren berechnen
    y_train_pred = clf.predict_proba(X_train)
    y_test_pred = clf.predict_proba(X_test)
    # F1 Score berechnen
    train_f1 = f1_score(y_train, y_train_pred.argmax(axis=1), average="macro")
    test_f1 = f1_score(y_test, y_test_pred.argmax(axis=1), average="macro")
    accuracy = clf.score(X_test, y_test)
    # Metriken dem Run hinzufügen
    run["test/accuracy"].append(accuracy)
    run["train/f1"].append(train_f1)
    run["test/f1"].append(test_f1)

# Modell speichern
dump(clf, "model.pkl")

# Modell dem Run hinzufügen
run["model"].upload("model.pkl")

# Run abschließen
run.stop()
