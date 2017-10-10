import pandas as pd

# Load the train and test datasets to create two DataFrames 
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Create a copy of test: test_one
#print test.head()
test_one = test.copy()

# Initialize a Survived column to 0
test_one['Survived'] = 0


# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
test_one['Survived'][test_one["Sex"]=="female" ]   = 1
 