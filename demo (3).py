import numpy as np
import pandas as pd

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
combine=[train,test]

train=train.drop(columns=['PassengerId','Ticket','Cabin'],inplace=False)
test=test.drop(columns=['PassengerId','Ticket','Cabin'],inplace=False)

train.info()
train.describe(include=['O'])
train['Embarked']=train['Embarked'].fillna('S').map({'C':0,'S':1,'Q':2})
train['Sex']=train['Sex'].map({'female':0,'male':1})
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'] = train['Title'].replace(['Lady','Mlle','Ms'], 'Miss')
train['Title'] = train['Title'].replace(['Countess','Mme','Dona'], 'Mrs')
train['Title'] = train['Title'].replace(['Sir','Capt','Col','Don','Dr','Major','Rev','Jonkheer'], 'Mr')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4}
train['Title'] = train['Title'].map(title_mapping)

test.info()
test.describe(include=['O'])
test['Fare'].fillna(test.Fare.median(),inplace=True)
test['Embarked']=test['Embarked'].fillna('S').map({'C':0,'S':1,'Q':2})
test['Sex']=test['Sex'].map({'female':0,'male':1})
test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Title'].replace(['Lady','Mlle','Ms'], 'Miss')
test['Title'] = test['Title'].replace(['Countess','Mme','Dona'], 'Mrs')
test['Title'] = test['Title'].replace(['Sir','Capt','Col','Don','Dr','Major','Rev','Jonkheer'], 'Mr')
test['Title'] = test['Title'].map(title_mapping)

set(train.Title)
pd.crosstab(test['Title'], test['Sex'])

train=train.drop(columns=['Name'],inplace=False)
test=test.drop(columns=['Name'],inplace=False)

train['withsomeone']=1
train['withsomeone'][train['SibSp']+train['Parch']==0]=0
test['withsomeone']=1
test['withsomeone'][test['SibSp']+test['Parch']==0]=0

train=train.drop(columns=['SibSp','Parch'],inplace=False)
test=test.drop(columns=['SibSp','Parch'],inplace=False)

medians=np.zeros(4)
for j in range(0, 4):
                    guess_df = pd.DataFrame(train[(train['Title'] == j+1)]['Age'].dropna())
                    medians[j] = int( guess_df.median()/0.5 + 0.5 ) * 0.5
for j in range(0, 4):
                    train.loc[ (train.Age.isnull()) & (train.Title == j+1),'Age'] = medians[j]
for j in range(0, 4):
                    guess_df = test[(test['Title'] == j+1)]['Age'].dropna()
                    medians[j] = int( guess_df.median()/0.5 + 0.5 ) * 0.5
for j in range(0, 4):
                    test.loc[ (test.Age.isnull()) & (test.Title == j+1),'Age'] = medians[j]
test.info()


plt.hist(train.Age,bins=30)

plt.hist(train.Fare,bins=50)

train['AgeBand'] = pd.cut(train['Age'], 5)
test['AgeBand'] = pd.cut(test['Age'], 5)
train.loc[ train['Age'] <= 16, 'Age'] = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[ train['Age'] > 64, 'Age']=4
test.loc[ test['Age'] <= 16, 'Age'] = 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1
test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2
test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3
test.loc[ test['Age'] > 64, 'Age']=4

train['Age']=train['Age'].astype(int)
test['Age']=test['Age'].astype(int)
train=train.drop(columns=['AgeBand'],inplace=False)
test=test.drop(columns=['AgeBand'],inplace=False)

train['FareBand'] = pd.cut(train['Fare'], 4)
test['FareBand'] = pd.cut(test['Fare'], 4)
test.loc[ test['Fare'] <= 7.91, 'Fare'] = 0
test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1
test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare']   = 2
test.loc[ test['Fare'] > 31, 'Fare'] = 3
test['Fare'] = test['Fare'].astype(int)
train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0
train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2
train.loc[ train['Fare'] > 31, 'Fare'] = 3
train['Fare'] = train['Fare'].astype(int)

train=train.drop(columns=['FareBand'],inplace=False)
test=test.drop(columns=['FareBand'],inplace=False)


trainy=train[['Survived']]
trainx=train.drop(columns=['Survived'],inplace=False)
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(trainx, trainy)

sb=random_forest.predict(test)
df=pd.read_csv('../input/test.csv')
df=df[['PassengerId']]
df['Survived']=sb

from IPython.display import FileLink
df.to_csv('submission_sample.csv',index=False)
FileLink('submission_sample.csv')
