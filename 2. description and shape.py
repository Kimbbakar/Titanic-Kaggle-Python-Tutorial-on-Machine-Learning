import pandas as pd

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv("train.csv")

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv("test.csv")
 

# .describe will give us some common observation on a topic
# mean, median, max, min, std, 
print (train.describe() )
print (train.shape )