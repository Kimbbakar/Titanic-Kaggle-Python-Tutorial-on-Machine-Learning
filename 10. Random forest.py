# Import the Numpy library
import pandas as pd
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn import tree

# Load the train and test datasets to create three DataFrames 
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
 
 
# Convert the male and female groups to integer form
train.loc[train["Sex"] == "male",'Sex' ] = 0
train.loc[train["Sex"] == "female",'Sex' ] = 1


# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")


# Convert the Embarked classes to integer form
train.loc[ train["Embarked"]=='S', "Embarked" ] = 0
train.loc[ train["Embarked"]=='C', "Embarked" ] = 1
train.loc[ train["Embarked"]=='Q', "Embarked" ] = 2
  

# Print the train data to see the available features

train.Age =  train.Age.fillna( np.mean(train["Age" ]) ) 
 
# Create "family_size" feature from "SibSp" and "Parch"
train["family_size"] = train.SibSp + train.Parch+1

print train.family_size
  


# Create the target and features numpy arrays: target, features_one
target = np.array(train["Survived" ].values )
features_three = np.array((train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch","family_size"]].values))
max_depth = 10
min_samples_split = 5
 

# # Fit your first decision tree: my_tree_one
my_tree_three = tree.DecisionTreeClassifier  (max_depth=max_depth,min_samples_split=min_samples_split,random_state=1 )
my_tree_three = my_tree_three.fit(features_three,target)

print(my_tree_three.feature_importances_)
print(my_tree_three.score(features_three, target))


# Convert the male and female groups to integer form
test.loc[test["Sex"] == "male",'Sex' ] = 0
test.loc[test["Sex"] == "female",'Sex' ] = 1
 

test.Age =  test.Age.fillna(test.Age.mean() )
test.set_value(152,'Fare' ,test.Fare.median() )  
#Data featuring for test data
test["family_size"] = test.SibSp + test.Parch+1


#alternate way to set value. test.Fare[152] =test.Fare.median()

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch","family_size"]].values


# Make your prediction using the test set
my_prediction = my_tree_three.predict(test_features)


# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])


# # Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_three.csv", index_label = ["PassengerId"])



