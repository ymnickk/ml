import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/fire.csv")
df[1] = [10]
df[2] = [10]
df[3] = [10]
df[4] = [10]
df[5] = [10]
df[6] = [1]
df[7] = [10]
df = df.drop(["Unnamed: 0"], axis=1)
X = df.drop('Fire Alarm_Yes', axis=1).values
y = df['Fire Alarm_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)