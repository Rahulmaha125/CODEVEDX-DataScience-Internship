import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("Advertising.csv")

print("========== ADVERTISING DATASET ==========")
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)


X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\n========== MODEL RESULT ==========")

print("\nR2 Score:")
print(round(r2_score(y_test, y_pred), 4))

print("\nMean Absolute Error:")
print(round(mean_absolute_error(y_test, y_pred), 2))

print("\nMean Squared Error:")
print(round(mean_squared_error(y_test, y_pred), 2))


sample = pd.DataFrame({
    "TV": [230.1],
    "Radio": [37.8],
    "Newspaper": [69.2]
})

prediction = model.predict(sample)

print("\n========== SAMPLE PREDICTION ==========")
print(f"Predicted Sales: {prediction[0]:.2f}")