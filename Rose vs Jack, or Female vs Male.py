import pandas as pd

# Load the train and test datasets to create two DataFrames 
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#print (train.head())

# Passengers that survived vs passengers that passed away
print (abs(train["Survived"].value_counts().iloc[0]-train["Survived"].value_counts().iloc[1] ) )

# As proportions
demo = train["Survived"].value_counts(normalize = True );
print ( demo  )


# Males that survived vs males that passed away
demo = train["Survived"][train["Sex"]=='male' ] .value_counts();
print (demo )

# Females that survived vs Females that passed away
demo = train["Survived"][train["Sex"]=='female' ] .value_counts();
print (demo )


# Normalized male survival
demo = train["Survived"][train["Sex"]=='male' ].value_counts(normalize = True )
print ( demo     )

# Normalized female survival
demo = train["Survived"][train["Sex"]=='female' ].value_counts(normalize = True )
print ( demo    )

