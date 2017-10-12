# Import the Numpy library
import pandas as pd
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn import tree

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# Print the train data to see the available features
print(train.head())

# Create the target and features numpy arrays: target, features_one
target = np.array(train["Survived" ].values )
features_one = np.array(train[["Pclass", "Sex", "Age", "Fare"]].values)

