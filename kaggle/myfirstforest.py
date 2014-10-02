""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September 2012
Revised: 15 April 2014
please see packages.python.org/milk/randomforests.html for more

""" 
import pandas as pd
import numpy as np
import csv as csv
import re
from sklearn.ensemble import RandomForestClassifier

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.


# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

# Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
# Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
# train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int
train_df = pd.concat([train_df, pd.get_dummies(train_df['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

#Custom Features
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"]
train_df["AgeClass"] = train_df["Age"]*train_df["Pclass"]

#Title feature:
train_df["Title"] = train_df['Name'].map(lambda x: re.search('.*, (.*?\.).*',x).group(1))
train_df = pd.concat([train_df, pd.get_dummies(train_df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)
#Grouping and examining title:
# gb = df.groupby(["Survived","title"])
# gb.title.count()[0]
# gb.title.count()[1]


# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Embarked', 'Title'], axis=1) 

# TEST DATA
test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
# if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
#     test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# # Again convert all Embarked strings to int
# test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

#commenting out the above in favor of using one hot encoding for embarked
test_df = pd.concat([test_df, pd.get_dummies(test_df['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)



# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

#Custom features
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"]
test_df["AgeClass"] = test_df["Age"]*test_df["Pclass"]

#Title features using one-hot encoding:
test_df["Title"] = test_df['Name'].map(lambda x: re.search('.*, (.*?\.).*',x).group(1))
test_df = pd.concat([test_df, pd.get_dummies(test_df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)


# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Embarked', 'Title'], axis=1) 

#Since the same titles dont necessarily exist in both the training set and the test set we will only take the columns that are in common between the two
common_cols = [col for col in set(train_df.columns).intersection(test_df.columns)]
test_df = test_df[common_cols]
common_cols.insert(0,'Survived')
train_df = train_df[common_cols]

# print test_df	
# print train_df

# print list(test_df)
# print list(train_df)

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values


print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)


predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
