# cricketanalysis001
#import libs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#dataset imported
data = pd.read_csv('player_data.csv')


#Data splitting
X = data[['Age', 'Experience', 'Position', 'Previous Performance']]
y = data['Player Performance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#encoding
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

#model training
model = LinearRegression()
model.fit(X_train_encoded, y_train)
y_pred = model.predict(X_test_encoded)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R-squared:', r2)
new_data = pd.DataFrame({'Age': [25], 'Experience': [5], 'Position': ['Forward'], 'Previous Performance': [0.85]})
new_data_encoded = pd.get_dummies(new_data)

prediction = model.predict(new_data_encoded)
print('Predicted Player Performance:', prediction)

