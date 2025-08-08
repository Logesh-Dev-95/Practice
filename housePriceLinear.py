import pandas as pd

# Data
data = {
    "Square_Feet": [650, 800, 1000, 1200, 1500, 1700, 2000, 2200, 2500, 2700,
                    3000, 3200, 3500, 3700, 4000],
    "Price": [70000, 85000, 110000, 125000, 150000, 165000, 200000, 210000, 
              250000, 260000, 300000, 320000, 360000, 370000, 400000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv("house_prices.csv", index=False)

print("house_prices.csv created successfully!")

import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('house_prices.csv')

df.head()

X = pd.DataFrame(df['Square_Feet'])
y = df['Price']

print(type(X))
print(type(y))

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
df.corr() #so the data set is good to perform linear regression

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train) 
print(type(X_train))

X_test = scalar.transform(X_test)
X_test


from sklearn.linear_model import LinearRegression

regression = LinearRegression(n_jobs=-1)

model = regression.fit(X_train,y_train)

y_predict = model.predict(X_test)


from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

print('mean squared error', mean_squared_error(y_predict,y_test))
print('root mean squared error',root_mean_squared_error(y_test,y_predict))
print('r2_score',r2_score(y_test,y_predict))


# mean squared error 31921398.550561406
# root mean squared error 5649.902525757538
# r2_score 0.9975923055751065