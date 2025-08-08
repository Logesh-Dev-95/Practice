# import pandas as pd
# import numpy as np

# np.random.seed(42)  # for reproducibility

# # Hours studied
# hours = np.linspace(1.5, 8, 25)  # 25 points from 1.5 to 8 hours

# # True underlying curve: quadratic relationship + noise
# # Base formula: score = -0.5*(hours-6)^2 + 95 (inverted parabola peaking at 6 hrs)
# scores = -0.5 * (hours - 6)**2 + 95  

# # Add random noise (±3 points)
# noise = np.random.normal(0, 3, size=len(hours))
# scores_noisy = scores + noise

# # Create DataFrame
# df = pd.DataFrame({
#     "Hours_Studied": hours,
#     "Exam_Score": np.round(scores_noisy, 1)  # round to 1 decimal
# })

# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

# X = df[['Hours_Studied']]
# y = df['Exam_Score']

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# from sklearn.preprocessing import PolynomialFeatures

# poly = PolynomialFeatures(degree=2)

# X_train_poly = poly.fit_transform(X_train)
# X_test_poly = poly.transform(X_test)

# from sklearn.linear_model import LinearRegression

# model = LinearRegression()

# model.fit(X_train_poly,y_train)

# y_pred = model.predict(X_test_poly)

# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

# print('r2_score',r2_score(y_test,y_pred))
# print('mean_absolute_error',mean_absolute_error(y_test,y_pred))
# print('mean_squared_error',mean_squared_error(y_test,y_pred))
# print('root_mean_squared_error',root_mean_squared_error(y_test,y_pred))

# # r2_score 0.09168655896140421
# # mean_absolute_error 1.7809156747951846
# # mean_squared_error 4.296685901488974
# # root_mean_squared_error 2.0728448811932294

# plt.scatter(X, y, color='blue', label='Actual Data')
# X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
# plt.plot(X_range, model.predict(poly.transform(X_range)), color='red', label='Polynomial Fit')
# plt.xlabel('Hours Studied')
# plt.ylabel('Exam Score')
# plt.legend()
# plt.show()



import pandas as pd
import numpy as np

np.random.seed(123)  # reproducibility

# Speeds from 10 to 120 km/h
speed = np.linspace(10, 120, 30)

# True underlying curve (inverted parabola peaking at ~55 km/h)
mpg = -0.02 * (speed - 55)**2 + 35

# Add realistic noise ±2 MPG
noise = np.random.normal(0, 2, size=len(speed))
mpg_noisy = mpg + noise

# Create DataFrame
df = pd.DataFrame({
    "Speed_kmh": speed,
    "MPG": np.round(mpg_noisy, 2)
})



import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X = df[['Speed_kmh']]
y = df['MPG']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train_poly,y_train)

y_pred = model.predict(X_test_poly)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

print('r2_score',r2_score(y_test,y_pred))
print('mean_absolute_error',mean_absolute_error(y_test,y_pred))
print('mean_squared_error',mean_squared_error(y_test,y_pred))
print('root_mean_squared_error',root_mean_squared_error(y_test,y_pred))



plt.scatter(X, y, color='blue', label='Actual Data')
X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
plt.plot(X_range, model.predict(poly.transform(X_range)), color='red', label='Polynomial Fit')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.show()



# r2_score 0.9901306141748947
# mean_absolute_error 2.3677606981360793
# mean_squared_error 6.063382330948742
# root_mean_squared_error 2.4623936181993207
