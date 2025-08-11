import pandas as pd

df = pd.DataFrame({
    'Age': [22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65],
    'Salary': [25000, 30000, 32000, 35000, 37000, 40000, 45000, 48000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000],
    'Bought': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
})


X = df[['Age','Salary']]
y = df['Bought']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(n_jobs=-1)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print('confusion_matrix',confusion_matrix(y_test,y_pred))
print('classification_report',classification_report(y_test,y_pred))


# confusion_matrix [[3 0]
#                   [0 1]]
# classification_report               precision    recall  f1-score   support

#            0       1.00      1.00      1.00         3
#            1       1.00      1.00      1.00         1

#     accuracy                           1.00         4
#    macro avg       1.00      1.00      1.00         4
# weighted avg       1.00      1.00      1.00         4
