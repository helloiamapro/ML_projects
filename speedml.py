import numpy as np
from sklearn.linear_model import LinearRegression

# Function to determine which plane's speed to reduce
def decide_speed_reduction(distance1, distance2, speed1, speed2):
    
    # Prepare the dataset for linear regression
    X = np.array([[distance1, speed1], [distance2, speed2]])
    y = np.array([0, 1])  # Binary target: 0 means reduce the speed of aircraft 1, 1 for aircraft 2

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X, y)

    # Predicting which aircraft's speed to reduce
    predicted_reduction = model.predict([[distance1, speed1], [distance2, speed2]])

    # Check which aircraft has a higher predicted value
    if predicted_reduction[0] < predicted_reduction[1]:
        # Reduce speed of aircraft 1
        new_speed1 = speed1 * 0.8  # Reduce speed by 20%
        new_speed2 = speed2  # Keep speed of aircraft 2 unchanged
    else:
        # Reduce speed of aircraft 2
        new_speed1 = speed1  # Keep speed of aircraft 1 unchanged
        new_speed2 = speed2 * 0.8  # Reduce speed by 20%

    return new_speed1, new_speed2
