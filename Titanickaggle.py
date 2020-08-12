import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df=pd.read_csv('C:\\Users\\SARAH\\\\train (1).csv')
df.info()
df.isnull().sum()
#Cabin and Age have null values, Embarked also has 2 null values
#Lets do exploratory data analysis

#What is the relationship between the pclass and survival
sns.barplot(x= "Pclass", y="Survived", data= df)
#we have more survivors in Class 1
#lets check for Emabrked ans sex
sns.barplot(x= "Embarked", y="Survived", data= df)
sns.barplot(x= "Sex", y="Survived", data= df)
#more females survived and people who embarked from C survived more

#Lets do some feature engneering, the Cabin column has a lot of missing values, we are dropping it
df.drop(['Cabin','PassengerId'],1,inplace=True)

#we need to fill in mising values, for the Age column, we will fill with the median by class and sex
age_groupby = df.groupby(['Sex','Pclass']).median()['Age']
#we  have the various medians, now i will fill the Age column by it
df['Age'] = df.groupby(['Sex','Pclass'])['Age'].apply(lambda x:x.fillna(x.median()))
#the apply method is used to apply a function to a whole column, so grouping the Age column by Sex and Pclass
#we fill in with the median
df['Age'].isnull().sum()
#filling the Embarked column by the most frequent, i can use simple imputer or simply use pandas fillna
most_frequent=df['Embarked'].mode()
df['Embarked'] =df['Embarked'].fillna('most_frequent')
df['Embarked'].isnull().sum()

#med_fare = df.groupby(['Pclass','Parch','SibSp']).Fare.median()[3][0][0]... 
#the Parch column and SibSp column make no sense on their own, lets use it to create the family size column
df['Family Size'] = df['SibSp'] + df['Parch']+ 1 #this means a person's sibling/spouse + parent or child(ren) + him/herself
# wether the person is a miss, Mrs will be extracted from the name column
df['Title'] = df['Name'].str.split(',', expand =True)[1].str.split('.',expand = True)[0]
#we also need to take the amount of fare paid per person
df['Fare_per_1'] = df['Fare']/df['Family Size']
#dropping the columns i dont need anymore
df.drop(['Ticket','Name','Parch','SibSp','Fare'],axis=1, inplace = True)

#using label encoder for the categorical features
cat_features = [features for features in df.columns if df[features].dtype=='object']
df.dtypes

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in cat_features:
    le.fit(list(df[i].values))
    df[i] = le.transform(list(df[i].values))

#lets use a heatmap to visualize correlation between features
plt.subplots(figsize=(12,8))
sns.heatmap(df.corr(), annot =True, cmap = 'Greens')


#carrying out scaling and PCA to check explained variance by each feature
X=df.iloc[:,1:]
y= df['Survived']


#splitting the training data to train and validation data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train,y_val = train_test_split(X,y, test_size=0.25, random_state=0)

#scaling and checking feature importances by PCA
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_val = sc_X.transform(X_val)


from sklearn.decomposition import PCA
pca = PCA(n_components=None, svd_solver='full')
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)
explained_variance= pca.explained_variance_ratio_ 
#feature 6 and 7 have little explaind variance

#dropping the title column and removing scaling seem to improve accuracy
X=df.iloc[:,1:]
y= df['Survived']
X.drop(['Title'],axis = 1, inplace = True)

#splitting the training data to train and validation data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train,y_val = train_test_split(X,y, test_size=0.25, random_state=0)


from sklearn.ensemble import GradientBoostingClassifier
clf =GradientBoostingClassifier()
clf.fit(X_train,y_train)

pred= clf.predict(X_val)
from sklearn.metrics import accuracy_score
accurate=accuracy_score(y_val,pred)

#checking for underfitting or overfitting
train_score=clf.score(X_train,y_train)
val_score= clf.score(X_val,y_val)


#Now using all the provided training data and testing data
#i will first apply same feature engineering to the test data
test=pd.read_csv('C:\\Users\\SARAH\\\\test (1).csv')
X_test=test.iloc[:,:]
X_test.drop(['Cabin','PassengerId'],1,inplace=True)
X_test.isnull().sum()

X_test['Age'] = X_test.groupby(['Sex','Pclass'])['Age'].apply(lambda x:x.fillna(x.median()))
X_test['Age'].isnull().sum()
#Fare has one missing value, i will fill with most frequent
X_test['Fare'] = X_test['Fare'].fillna(X_test['Fare'].mode()[0])


X_test['Family Size'] = X_test['SibSp'] + X_test['Parch']+ 1 
X_test['Fare_per_1'] = X_test['Fare']/X_test['Family Size']
X_test.drop(['Ticket','Name','Parch','SibSp','Fare'],axis=1, inplace = True)


#using label encoder for the categorical features
cat_features2 = [features for features in X_test.columns if X_test[features].dtype=='object']
X_test.dtypes

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in cat_features2:
    le.fit(list(X_test[i].values))
    X_test[i] = le.transform(list(X_test[i].values))

X_train = X
y_train=y
X_test.isnull().sum()
from sklearn.ensemble import GradientBoostingClassifier
clf =GradientBoostingClassifier()
clf.fit(X_train,y_train)

y_pred= clf.predict(X_test)


test2=pd.read_csv('C:\\Users\\SARAH\\\\test (1).csv')

submission2= pd.DataFrame()
submission2['PassengerId'] = test2['PassengerId']
submission2['Survived'] = y_pred
submission.to_csv('Predictions2.csv', index=False)