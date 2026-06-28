import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("Titanic-Dataset.csv")


print("========== TITANIC DATASET ==========")
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())




df["Age"].fillna(df["Age"].median(), inplace=True)


df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)


df.drop("Cabin", axis=1, inplace=True)


df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})


X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULT ==========")
print("\nAccuracy Score:")
print(round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


sample = pd.DataFrame({
    "Pclass": [3],
    "Sex": [0],
    "Age": [22],
    "SibSp": [1],
    "Parch": [0],
    "Fare": [7.25],
    "Embarked": [0]
})

prediction = model.predict(sample)

print("\n========== SAMPLE PREDICTION ==========")

if prediction[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger Did Not Survive")