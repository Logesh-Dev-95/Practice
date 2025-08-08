import pandas as pd

# Data
data = {
    "Hours_Studied": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 
                      5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
    "Exam_Score": [52, 55, 60, 62, 67, 72, 75, 78, 
                   82, 88, 90, 94, 96, 98]
}

df = pd.DataFrame(data)

df.head()


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

X = df[['Hours_Studied']]
print(type(X))
y = df['Exam_Score']
print(type(y))

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.25)

from sklearn.linear_model import LinearRegression

Regression = LinearRegression(n_jobs=-1)

model = Regression.fit(X_train,y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error

print("r2_score",r2_score(y_test,y_pred))
print("mean_squared_error",mean_squared_error(y_test,y_pred))
print("root_mean_squared_error",root_mean_squared_error(y_test,y_pred))


# r2_score 0.9918413234802979
# mean_squared_error 2.600578140655052
# root_mean_squared_error 1.6126308135016683