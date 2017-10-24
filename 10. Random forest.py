import pandas as pd
import numpy as np
 
# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# Load the train and test datasets to create four DataFrames 
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
 

# Create the target and features numpy arrays: target, features_four
target = np.array(train["Survived" ].values )
features_four = np.array((train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch","family_size"]].values))

# Using Random Forest Classifier
forest = RandomForestClassifier  (max_depth=10,min_samples_split=2,n_estimators=100 , random_state=1 )
my_tree_four = forest.fit(features_four,target)

print(my_tree_four.feature_importances_)
print(my_tree_four.score(features_four, target))


# # Convert the male and female groups to integer form
test.loc[test["Sex"] == "male",'Sex' ] = 0
test.loc[test["Sex"] == "female",'Sex' ] = 1
 

test.Age =  test.Age.fillna(test.Age.mean() )
test.Fare[152] =test.Fare.median() 
#Data featuring for test data
test["family_size"] = test.SibSp + test.Parch+1




# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch","family_size"]].values


# Make your prediction using the test set
my_prediction = my_tree_four.predict(test_features)
 

PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])


my_solution.to_csv("my_solution_four.csv", index_label = ["PassengerId"])



