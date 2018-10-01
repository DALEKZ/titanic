#!/usr/bin/python
#coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from interval import Interval
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#Age缺失处理
def set_missing_ages(train):
    age_df = train[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y = known_age[:, 0]
    x = known_age[:,1:]     #特征值'Fare', 'Parch', 'SibSp', 'Pclass'
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x,y)
    predictAges = rfr.predict(unknown_age[:, 1::])
    train.loc[(train.Age.isnull(), 'Age')] = predictAges
    return train, rfr

#Cabin处理
def set_Cabin_type(train):
    train.loc[(train.Cabin.notnull()), 'Cabin'] = 'Yes'
    train.loc[(train.Cabin.isnull()), 'Cabin'] = 'No'
    return train

def getTitle(name):
    str = name.split(',')[1]
    title = str.split('.')[0]
    title = title.strip()
    return title

train, rfr = set_missing_ages(train)
train = set_Cabin_type(train)

title_df = pd.DataFrame()
title_df['Title'] = train['Name'].map(getTitle)
title_df.loc[title_df.Title=='Mlle','Title'] = 'Miss'
title_df.loc[title_df.Title=='Major','Title'] = 'Others'
title_df.loc[title_df.Title=='Lady','Title'] = 'Others'
title_df.loc[title_df.Title=='Jonkheer','Title'] = 'Master'
title_df.loc[title_df.Title=='Don','Title'] = 'Others'
title_df.loc[title_df.Title=='Mme','Title'] = 'Mrs'
title_df.loc[title_df.Title=='Capt','Title'] = 'Others'
title_df.loc[title_df.Title=='the Countess','Title'] = 'Others'
title_df.loc[title_df.Title=='Sir','Title'] = 'Others'

train['Name'] = title_df['Title']
print(train['Name'])
bin = [0,12,19,65,80]
train['Age_group'] = pd.cut(train['Age'],bin)
train['Age_group'] = train['Age_group'].astype('str',copy=True)
train.loc[(train.Age_group == '(0, 12]'),'Age_group'] = '1'
train.loc[(train.Age_group == '(12, 19]'),'Age_group'] = '2'
train.loc[(train.Age_group ==  '(19, 65]'),'Age_group'] = '3'
train.loc[(train.Age_group == '(65, 80]'),'Age_group'] = '4'
train['Age_group'] = train['Age_group'].astype('int',copy=True)
train['Child'] = None
train.loc[(train.Age_group == 1),'Child'] = 'yes'
train.loc[(train.Age_group != 1),'Child'] = 'no'

#变量变换
dummy_Cabin = pd.get_dummies(train['Cabin'], prefix='Cabin')
dummy_Name = pd.get_dummies(train['Name'], prefix='Name')
dummy_Child = pd.get_dummies(train['Child'],prefix="Child")
dummy_Embarked = pd.get_dummies(train['Embarked'], prefix='Embarked')
dummy_Sex = pd.get_dummies(train['Sex'], prefix='Sex')
dummy_Pclass = pd.get_dummies(train['Pclass'], prefix='Pclass')
train_new = pd.concat([train, dummy_Cabin,dummy_Name, dummy_Embarked, dummy_Sex, dummy_Pclass,dummy_Child],axis=1)
train_new.drop(['Cabin','Name','Embarked', 'Sex', 'Pclass', 'Ticket'], axis=1, inplace=True)

#Scaling Age & Fare
import sklearn.preprocessing as prerocessing
scaler = prerocessing.StandardScaler()
#train_new['Age_scaled'] = scaler.fit_transform(train_new['Age'].values.reshape(-1,1))
train_new['Fare_scaled'] = scaler.fit_transform(train_new['Fare'].values.reshape(-1,1))
train_new.drop(['Age', 'Fare'], axis=1, inplace=True)
from sklearn import linear_model, cross_validation

train_df = train_new.filter(regex='Survived|Age_.*|SibSp|Name_.*|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Child_.*')
train_np = train_df.as_matrix()
y = train_np[:, 0]
x = train_np[:,1:]


clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
clf.fit(x,y)


#交叉验证
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf,x,y,cv=5)
print(scores.mean())
#
# split_train, split_cv = cross_validation.train_test_split(train,test_size=0.3,random_state=0)
# train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*')
# clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
# clf.fit(train_df.as_matrix()[:,1:], train_df.as_matrix()[:,0])
#
# cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*')
# prediction = clf.predict(cv_df.as_matrix()[:,1:])
# origin_train = pd.read_csv('data/train.csv')
# bad_cases = origin_train.loc[origin_train['PassengerId'].isin(split_cv[prediction != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
# print('badcase:',bad_cases)


#test集处理
test.loc[(test.Fare.isnull()),'Fare'] = 0
age_df2 = test[['Age', 'Fare','Parch', 'SibSp', 'Pclass']]
null_age = age_df2[test.Age.isnull()].as_matrix()

x = null_age[:,1:]
predictedAges = rfr.predict(x)
test.loc[(test.Age.isnull()), 'Age'] = predictedAges

test_title_df = pd.DataFrame()
test_title_df['Title'] = test['Name'].map(getTitle)
test_title_df.loc[test_title_df.Title=='Dona','Title'] = 'Others'
test['Name'] = test_title_df['Title']


bin = [0,12,19,65,80]
test['Age_group'] = pd.cut(test['Age'],bin)
test['Age_group'] = test['Age_group'].astype('str',copy=True)
test.loc[(test.Age_group == '(0, 12]'),'Age_group'] = '1'
test.loc[(test.Age_group == '(12, 19]'),'Age_group'] = '2'
test.loc[(test.Age_group ==  '(19, 65]'),'Age_group'] = '3'
test.loc[(test.Age_group == '(65, 80]'),'Age_group'] = '4'
test['Age_group'] = test['Age_group'].astype('int',copy=True)
test['Child'] = None
test.loc[(test.Age_group == 1),'Child'] = 'yes'
test.loc[(test.Age_group != 1),'Child'] = 'no'

test = set_Cabin_type(test)
dummy_Cabin = pd.get_dummies(test['Cabin'],prefix='Cabin')
dummy_Name = pd.get_dummies(test['Name'], prefix='Name')
dummy_Embarked = pd.get_dummies(test['Embarked'],prefix='Embarked')
dummy_Child = pd.get_dummies(test['Child'],prefix="Child")
dummy_Sex = pd.get_dummies(test['Sex'],prefix='Sex')
dummy_Pclass = pd.get_dummies(test['Pclass'], prefix='Pclass')
test_new = pd.concat([test,dummy_Cabin,dummy_Name,dummy_Embarked, dummy_Sex, dummy_Pclass,dummy_Child],axis=1)
test_new.drop(['Pclass', 'Sex','Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
#test_new['Age_scaled'] = scaler.fit_transform(test_new['Age'].values.reshape(-1,1))
test_new['Fare_scaled'] = scaler.fit_transform(test_new['Fare'].values.reshape(-1,1))
test_new.drop(['Age', 'Fare'], axis=1, inplace=True)

data_test = test_new.filter(regex='Age_.*|SibSp|Name_.*|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Child_.*')
prediction = clf.predict(data_test)
result = pd.DataFrame({'PassengerId':test['PassengerId'].as_matrix(), 'Survived':prediction.astype(np.int32)})
result.to_csv("data/titanic_pdt2.csv", index=False)



def image_show():
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    fig = plt.figure(figsize=(10,10))
    fig.set(alpha=0.2)

    plt.subplot2grid((3,3), (0,0))
    train.Survived.value_counts().plot(kind='bar')
    plt.title(u'获救情况')
    plt.ylabel(u'人数')

    plt.subplot2grid((3,3), (0,1))
    train.Pclass.value_counts().plot(kind='bar')
    plt.title(u'乘客等级分布')

    plt.subplot2grid((3,3),(0,2))
    plt.scatter(train.Survived, train.Age)
    plt.ylabel(u'年龄')

    plt.subplot2grid((3,3), (1,0), colspan=2)
    train.Age[train.Pclass == 1].plot(kind='kde')
    train.Age[train.Pclass == 2].plot(kind='kde')
    train.Age[train.Pclass == 3].plot(kind='kde')
    plt.ylabel(u'密度')
    plt.xlabel(u'年龄')
    plt.legend((u'头等舱',u'2等舱',u'3等舱'),loc='best')

    plt.show()
