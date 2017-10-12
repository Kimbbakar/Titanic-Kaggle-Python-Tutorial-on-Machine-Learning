# Import the Numpy library
import pandas as pd
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn import tree



# Load the train and test datasets to create two DataFrames 
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
 


 
# # Convert the male and female groups to integer form
# train["Sex"][train["Sex"] == "male"] = 0
# train["Sex"][train["Sex"] == "female"] = 1

# # Impute the Embarked variable
# train["Embarked"] = train["Embarked"].fillna("S")


# # Convert the Embarked classes to integer form
# train["Embarked"] [ train["Embarked"] =='S' ] = 0;  
# train["Embarked"] [ train["Embarked"] =='C' ] = 1;  
# train["Embarked"] [ train["Embarked"] =='Q' ] = 2;  


#Print the Sex and Embarked columns
print (train["Sex"]["Embarked"] )
#print (train.loc[ :, ["Sex","Embarked"] ]  )