import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/weather.csv")
numeric_features = df.select_dtypes(include=["float64", "int64"]).columns
df = df[numeric_features]

df = df.drop(["Unnamed: 0", "index", "index.1"], axis=1)
X = df.drop('RainTomorrow', axis=1).values
y = df['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)