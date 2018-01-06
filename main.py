# data analysis and wrangling
import pandas as pd # https://pandas.pydata.org/pandas-docs/stable/
import numpy as np
import random as rnd

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
combine = [train_df, test_df]

# Column names:
# ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']


# print((train_df.loc[(train_df['Parch'] == 0) & (train_df['SibSp'] == 0)]).shape)
# print((train_df.loc[train_df['Sex'] == 'male']).shape)

print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('-'*30)
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('-'*30)
print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# USEFUL pandas METHODS TO USE ON DATA FRAMES

# For each column, gives the count, mean, stdev, min, max, and percentile values (default is 25, 50, 75 but can
# specify your own)
# print(train_df.describe(include=[np.object]))

# Lists all the column names, along with the number of non-null entries for each and the datatype of each. It also
# tells you how many rows there are
# print(train_df.info())

# Lists all the column names for a dataset
# print(train_df.columns.values)
