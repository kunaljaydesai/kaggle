import pandas as pd
import numpy as np
import csv as csv
import random
from heapq import heappush, heappop

def print_full(train_df):
	pd.set_option('display.max_rows', len(train_df))
	print(train_df)

def distance_to(pt1, pt2):
	assert len(pt1) == len(pt2), "points aren't in the same dimension"
	squared_distance = 0
	for i in range(0, len(pt1)):
		squared_distance = squared_distance + (pt1[i] - pt2[i]) ** 2
	return squared_distance ** 0.5

def k_means(num, pt1, df):
	heap = []
	for index, row in df.iterrows():
		pt2 = [row['Pclass'], row['Age'], row['SibSp'], row['Parch'], row['Fare'], row['Embarked'], row['Gender']]
		heappush(heap, [distance_to(pt1, pt2), row['Survived']])
	closest_num = []
	for i in range(0, num):
		closest_num.append(heappop(heap))
	return np.mean([i[1] for i in closest_num])


train_df = pd.read_csv('train.csv', header=0)
train_df['Gender'] = train_df['Sex'].map({'female' : 0, 'male' : 1}).astype(int)

train_df.Embarked[train_df.Embarked.isnull()] = train_df.Embarked.dropna().mode().values

ports = list(enumerate(np.unique(train_df['Embarked'])))
ports_dict = {name : i for i, name in ports}
train_df.Embarked = train_df.Embarked.map(lambda x : ports_dict[x]).astype(int)

train_df.Age[train_df.Age.isnull()] = train_df.Embarked.dropna().median()

train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)



test_df = pd.read_csv('test.csv', header=0)
test_df['Gender'] = test_df['Sex'].map({'female' : 0, 'male' : 1}).astype(int)

test_df.Embarked[test_df.Embarked.isnull()] = test_df.Embarked.dropna().mode().values

ports = list(enumerate(np.unique(test_df['Embarked'])))
ports_dict = {name : i for i, name in ports}
test_df.Embarked = test_df.Embarked.map(lambda x : ports_dict[x]).astype(int)

test_df.Age[test_df.Age.isnull()] = test_df.Embarked.dropna().median()

test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

random_row_index = random.randint(0, len(test_df))
print("Random row index: " + str(random_row_index))
row = test_df.ix[random_row_index]
pt1 = [row['Pclass'], row['Age'], row['SibSp'], row['Parch'], row['Fare'], row['Embarked'], row['Gender']]
print("Probability that this person survived: " + str(k_means(10, pt1, train_df)))


