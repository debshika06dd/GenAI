import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"ford.csv")
df.head()
df.shape()
df.info()
df.describe()
df.isnull().sum()

sns.histplot(df['price'], bins = 50, kde=True)
sns.heatmap(df.corr(numeric_only = True), annot = True)
sns.boxplot(data = df, x = 'year', y = 'price')
plt.xticks(rotation = 90)
sns.scatterplot(data = df, x = 'mileage', y = 'price')
sns.boxplot(data = df, x = 'engineSize', y = 'price')
df.columnssns.boxplot(data = df, x = 'tranmission', y = 'price')
sns.boxplot(data, x = 'fuelTyp', y = 'price')
sns.boxplot(x = df['model'], y = df['price'])
plt.xticks(rotation = 90)
X = df.drop(columns = ['price'], axis = 1)
y = df['price']
X
df.columns
X_one_encode = X_one_encode.astype(int)
X_one_encode

from sklearn.preprocessing import LabelEncoder
columns = ['model', 'transmission', 'fuelType']

Xlabel = X.xopy() #make a safe copy
label_encoders = {}

for col in columns:
    le = LabelEncoder()
    Xlabel[col] = le.fit_transform(Xlabel[col].astype(str)) #convert to string in case of nulls
    label_encoders[col] = le

Xlabel

from sklearn.preprocessing import StandardScaler

numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
scaler = StandardScaler()
X_one_encode[numerical_cols] = scaler.fit_transform(X_one_encode[numerical_cols])

X_one_encode

Xlabel.columns

Xlabel[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']] = scaler.fit_transform(Xlabel[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']])

Xlabel

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_sqaured_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X_one_encode, y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.prediction(X_test)

y_pred

y_test

r2 = r2_score(y_test, y_pred)
r2

n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - ((1-r2) * (n-1)) / (n-p-1)
print("Adjusted R^2 Score: ", adjusted_r2)

X_train, X_test, y_train, y_test = train_test_split(Xlabel, y, test_size=0.33, random_state=42)

model2 = LinearRegression()
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)

y_pred

y_test
r2 = r2_score(y_test, y_pred)
r2



