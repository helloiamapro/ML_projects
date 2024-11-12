import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv('atcdata.csv')

X = df[['x_position', 'y_position', 'speed', 'direction', 'time']]
y = df[['next_x_position', 'next_y_position']] 

model = LinearRegression()


model.fit(X, y)


sample_input = [[150, 250, 5, 90, 6]]  
predicted_position = model.predict(sample_input)

print(predicted_position)
