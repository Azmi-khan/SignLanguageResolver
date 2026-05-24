import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

data = pd.read_csv("sign_language_data.csv")
print("loading the data matrix")
dataset = pd.read_csv("sign_language_data.csv")
print(f"data loaded successfully shape is{dataset.shape}")

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n training the model using Random forest")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("training complete")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n system accuracy is: {accuracy * 100:.2f}%")

print("\n Statistical Performance Matrix:")
print(classification_report(y_test, y_pred))

output_path = "sign_model.p"
with open(output_path, "wb") as f:
    pickle.dump(model, f)

print(f"model saved successfully to {output_path}")
