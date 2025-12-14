import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

df = pd.read_csv('AH_Excess_Deaths_by_Sex__Age__and_Race_and_Hispanic_Origin.csv')

columns_to_keep = ['Deaths (weighted)', 'RaceEthnicity', 'Sex', 'AgeGroup', 'MMWRyear', 'MMWRweek']

missing_cols = [col for col in columns_to_keep if col not in df.columns]
if missing_cols:
    print(f"Error: Missing columns: {missing_cols}")
    exit()

df = df[columns_to_keep]

df = df.dropna()

le_race = LabelEncoder()
df['RaceEthnicity'] = le_race.fit_transform(df['RaceEthnicity'].astype(str))

le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'].astype(str))

le_age = LabelEncoder()
df['AgeGroup'] = le_age.fit_transform(df['AgeGroup'].astype(str))

X = df.drop('Deaths (weighted)', axis=1)
y_reg = df['Deaths (weighted)']

mean_deaths = y_reg.mean()
y_class = (y_reg > mean_deaths).astype(int)
print(f"Mean Deaths: {mean_deaths:.2f}")

X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

print("\n--- Linear Regression (Predicting Exact Deaths) ---")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_reg_train)
y_pred_lin = lin_reg.predict(X_test)

mse_lin = mean_squared_error(y_reg_test, y_pred_lin)
r2_lin = r2_score(y_reg_test, y_pred_lin)
print(f"Mean Squared Error: {mse_lin:.2f}")
print(f"R2 Score: {r2_lin:.4f}")

print("\n--- Decision Tree Regressor (Predicting Exact Deaths) ---")
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train, y_reg_train)
y_pred_dt = dt_reg.predict(X_test)

mse_dt = mean_squared_error(y_reg_test, y_pred_dt)
r2_dt = r2_score(y_reg_test, y_pred_dt)
print(f"Mean Squared Error: {mse_dt:.2f}")
print(f"R2 Score: {r2_dt:.4f}")

print("\n--- Logistic Regression (Predicting High vs Low Deaths) ---")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_class_train)
y_pred_log = log_reg.predict(X_test)

accuracy_log = accuracy_score(y_class_test, y_pred_log)
print(f"Accuracy: {accuracy_log:.4f}")

import joblib

joblib.dump(dt_reg, 'decision_tree_model.pkl')
joblib.dump(le_race, 'le_race.pkl')
joblib.dump(le_sex, 'le_sex.pkl')
joblib.dump(le_age, 'le_age.pkl')
print("Model created and saved successfully.")