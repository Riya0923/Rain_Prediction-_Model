''' TABLE OF CONTENT :
    1. The Problem Statement
    2. Import Libraries
    3. Import Dataset
    4. Exploratory Data Analysis
       4.1. Preview the Dataset
       4.2. Dimension of Dataset
       4.3. Describing the attributes
       4.4. Finding all categorical and continuous values
       4.5. Exploring Unique values
       4.6. Finding Null Values
       4.7. Visualising Missing Values
       4.8. Dealing with the missing values
    5. Univariate Data Analysis
    
'''
"""
1. THE PROBLEM STATEMENT 
 
    This model will predict whether there will be rain tomorrow in Australia or not. In this model we will be using the Logistic Regression with Pythonand Scikit-Learn.

    We will train a binary classification model using Logistic Regression. We have used the Rain in Australia dataset for this project.

    The dataset contains approximately 10 years of daily weather observations from several locations across Australia.

    Here, RainTomorrow will be the target variable to predict.
    
 """
# 2. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import missingno as msn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pickle
import warnings

warnings.filterwarnings("ignore")

# 3. IMPORT DATASET

df1 = pd.read_csv('C:\Rain_Model\weatherAUS.csv')

# 4. EXPLORATORY DATA ANALYSIS

#    4.1.  Preview the Dataset

df1.head(10)

#    4.2. Dimension of Dataset

print(f'The number of rows are {df1.shape[0] } and the number of columns are {df1.shape[1]}')

#    4.3. Describing the attributes

df1.info()

#    4.4. Finding all the categorical and continuous values

categorical_col, contin_val=[],[]

for i in df1.columns:

    if df1[i].dtype == 'object':
        categorical_col.append(i)
    else:
        contin_val.append(i)

print(categorical_col)
print(contin_val)

#    4.5. Exploring Unique values

df1.nunique()

#    4.6. Finding Null Value

df1.isnull().sum()

#    4.7. Visualising Missing Values

msn.matrix(df1)

msn.bar(df1, sort='ascending')

msn.heatmap(df1)

"""The above graphs show that the number of missing values are high in: Sunshine, Evaporation, Cloud3pm and Cloud9am."""

plt.figure(figsize=(17,15))
ax = sns.heatmap(df1.corr(), square=True, annot=True, fmt='.2f')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()

''' Changing yes and no to 1 and 0 in some columns '''

df1['RainTomorrow'] = df1['RainTomorrow'].map({'Yes': 1, 'No': 0})
df1['RainToday'] = df1['RainToday'].map({'Yes':1,'No':0})
print(df1.RainToday)
print(df1.RainTomorrow)

#    4.8. Dealing with the missing values

#Checking percentage of missing data in every column

(df1.isnull().sum()/len(df1))*100

#Filling the missing values for continuous variables with mode

df1['RainTomorrow']=df1['RainTomorrow'].fillna(df1['RainTomorrow'].mode()[0])

#Checking percentage of missing data in every column

(df1.isnull().sum()/len(df1))*100

# 5. UNIVARIATE DATA ANALYSIS

#    5.1. Exploring 'RainTomorrow' target variable

df1['RainTomorrow'].unique()

df1['RainTomorrow'].isnull().sum()

''' Visualisation '''

df1['RainTomorrow'] = df1['RainTomorrow'].map({1:'Yes',0:'No'})
df1['RainToday'] = df1['RainToday'].map({1:'Yes',0:'No'})

f, ax = plt.subplots(1,2)
print(df1['RainToday'].value_counts())
print(df1['RainTomorrow'].value_counts())

plt.figure(figsize = (20,20))
sns.countplot(data = df1, x = 'RainToday',ax=ax[0])
sns.countplot(data = df1, x = 'RainTomorrow',ax = ax[1])

""" Viewing percentage of frequency of distribution of values """

df1['RainTomorrow'].value_counts()/len(df1)

""" Interpretation """

print(f"For the Rain Tomorrow Feature")
print(f"Yes is {0.780854*100}% times")
print(f"No is {0.219146*100}% times")

#    5.2. Conclusion of Univariate Analysis


'''    * The number of unique values in RainTomorrow variable is 2.
       *   The two unique values are No and Yes.
       *   Out of the total number of RainTomorrow values, No appears 78.08% times and Yes appears 21.91% times.
       *  The univariate plot confirms that :

              *   The No variable have 113583 entries, and
              *   The Yes variable have 31877 entries.
'''       

# 6. BIVARIATE ANALYSIS

#     6.1. Exploring categorical values

categorical = [var for var in df1.columns if df1[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)


# View Categorical Variables

df1[categorical].head()

"""
      Summary of categorical variables :

         *   There is a 'Date' variable which is denoted by Date column.
         *   Apart from that, there are 6 categorical variables :  Location, WindGustDir, WindDir9am, WindDir3pm, RainToday and RainTomorrow.
         *   There are two binary categorical variables - RainToday and RainTomorrow.
         *   Where, RainTomorrow is our target variable.

"""

#      Missing values in categorical variables

df1[categorical].isnull().sum()

""" 
     We can see that there are only 4 categorical variables which contains missing values :
     These are WindGustDir, WindDir9am, WindDir3pm and RainToday.

"""

#      The frequency count of categorical variables.

for var in categorical:

    print(df1[var].value_counts())

# view frequency distribution of categorical variables

for var in categorical:

    print(df1[var].value_counts()/np.float(len(df1)))

#       Checking for high cardinality 

'''  
     A high number of labels within a variable is known as high cardinality.
     Here, we can see 'Date' variable has the highest cardinality. Thus, it needs to be pre-processed.
     
'''

#       Feature Engineering of Date variable


df1['Date'].dtypes

""" Since, the datatype of the 'Date' variable is Object. Parse it into datetime format."""

df1['Date'] = pd.to_datetime(df1['Date'])

""" Extract year from Date."""

df1['Year'] = df1['Date'].dt.year

df1['Year'].head()

""" Extract month from Date."""

df1['Month'] = df1['Date'].dt.month

df1['Month'].head()

""" Extract day from Date."""

df1['Day'] = df1['Date'].dt.day

df1['Day'].head()

""" Review the dataset."""

df1.info()

"""
     We can observe three additional columns created from Date variable.
     Thus, drop the original Date variable from the dataset.
     
"""

df1.drop('Date', axis=1, inplace = True)

# Date variable has been successfully removed from the dataset
df1.head()

#        Explore all the categorical values one by one.

# find categorical variables

categorical = [var for var in df1.columns if df1[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)

# check for missing values in categorical variables

df1[categorical].isnull().sum()


#           Exploring 'Location' variable.

# print number of labels in Location variable

print('Location contains', len(df1.Location.unique()), 'labels')

# Labels in 'Location' variable
df1.Location.unique()

# check frequency distribution of values in Location variable

df1.Location.value_counts()

#           One Hot Encoding of 'Location' variable
"""

*    k-1 dummy variables
*   preview the dataset with head() method


"""

pd.get_dummies(df1.Location, drop_first=True).head()

#           Exploring 'WindGustDir' variable

# print number of labels in WindGustDir variable

print('WindGustDir contains', len(df1['WindGustDir'].unique()), 'labels')

# check labels in WindGustDir variable

df1['WindGustDir'].unique()

# check frequency distribution of values in WindGustDir variable

df1.WindGustDir.value_counts()

#           One Hot Encoding of 'WindGust' variable.
"""


*    k-1 dummy variables
*   preview the dataset with head() method


"""

pd.get_dummies(df1.WindGustDir, drop_first=True, dummy_na=True).head()

# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df1.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)

# Here, we can see that there are 10326 missing values in WindGustDir variable.

#            Explore WindDir9am variable.


#print number of labels in WindDir9am variable

print('WindDir9am contains', len(df1['WindDir9am'].unique()), 'labels')

# check labels in WindDir9am variable

df1['WindDir9am'].unique()

# check frequency distribution of values in WindDir9am variable

df1['WindDir9am'].value_counts()

#            One Hot Encoding of WindDir9am variable

# get k-1 dummy variables after One Hot Encoding
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df1.WindDir9am, drop_first=True, dummy_na=True).head()

# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df1.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)

#           Explore WindDir3pm variable

# print number of labels in WindDir3pm variable

print('WindDir3pm contains', len(df1['WindDir3pm'].unique()), 'labels')

# checking labels in WindDir3pm variable

df1['WindDir3pm'].unique()

# check frequency distribution of values in WindDir3pm variable

df1['WindDir3pm'].value_counts()

#            One Hot Encoding of WindDir3pm variable

# get k-1 dummy variables after One Hot Encoding
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df1.WindDir3pm, drop_first=True, dummy_na=True).head()

# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df1.WindDir3pm,drop_first=True, dummy_na=True).sum(axis=0)

#            Explore RainToday variable

# print number of labels in RainToday variable

print('RainToday contains', len(df1['RainToday'].unique()), 'labels')

# check labels in WindGustDir variable

df1['RainToday'].unique()

# check frequency distribution of values in WindGustDir variable

df1.RainToday.value_counts()

#             One Hot Encoding of RainToday variable

# get k-1 dummy variables after One Hot Encoding
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df1.RainToday, dummy_na=True).head()

# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df1.RainToday, dummy_na=True).sum(axis=0)

# There are 3261 missing values

#      6.2. EXPLORE NUMERICAL VALUES


#find the numerical variables

numerical = [var for var in df1.columns if df1[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are:',numerical)

#view the numericalvariables

df1[numerical].head()

""" 
         Summary of numerical variables :
            * There are 16 numerical variables.
            * These are given by MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am and Temp3pm.
            * All of the numerical variables are of continuous type

### Explore the problems within numerical variables

Missing values in numerical variables
"""

# check missing values in numerical variables

df1[numerical].isnull().sum()

"""We can see there are 16 numerical variables containing missing values

### Outliers in numerical variables
"""

#view summary statics in numerical variables
print(round(df1[numerical].describe()),2)

"""### On closer inspection, we can see that the Rainfall, Evaporation, WindSpeed9am and WindSpeed3pm columns may contain outliers"""

#draw the boxplots to visualize outliers in the above variables

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
fig = df1.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')

plt.subplot(2,2,2)
fig = df1.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')

plt.subplot(2,2,3)
fig = df1.boxplot(column = 'WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')

plt.subplot(2,2,4)
fig = df1.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')

''' 
        Check the distribution of variables

            * Now, we will plot the histograms to check distributions to find out if they are normal or skewed.

            * If the variable follows normal distribution, then we will do Extreme Value Analysis otherwise if they are skewed, I will find IQR (Interquantile range).
        
'''

#plot the histogram to check the distribution

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
fig = df1['Rainfall'].hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')

plt.subplot(2,2,2)
fig = df1['Evaporation'].hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')

plt.subplot(2,2,3)
fig = df1['WindSpeed9am'].hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')

plt.subplot(2,2,4)
fig = df1['WindSpeed3pm'].hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')

# Find the oultiers for Rainfall Variable

Q1 = df1['Rainfall'].describe()[4]
Q2 = df1['Rainfall'].describe()[6]
IQR = Q2 - Q1
Lower_fence = Q1 - (IQR * 3)
Upper_fence = Q2 + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

df1['Rainfall'].describe()

""" For Rainfall, the minimum and maximum values are 0.0 and 371.0. So, the outliers are values > 3.2."""

# Find the oultiers for Evaporation Variable

Q1 = df1['Evaporation'].describe()[4]
Q2 = df1['Evaporation'].describe()[6]
IQR = Q2 - Q1
Lower_fence = Q1 - (IQR * 3)
Upper_fence = Q2 + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

df1['Evaporation'].describe()

""" For Evaporation, the minimum and maximum values are 0.0 and 145.0. So, the outliers are values > 21.8."""

# Find the oultiers for WindSpeed9am Variable

Q1 = df1['WindSpeed9am'].describe()[4]
Q2 = df1['WindSpeed9am'].describe()[6]
IQR = Q2 - Q1
Lower_fence = Q1 - (IQR * 3)
Upper_fence = Q2 + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

df1['WindSpeed9am'].describe()

""" For WindSpeed9am, the minimum and maximum values are 0.0 and 130.0. So, the outliers are values > 55.0."""

# Find the oultiers for WindSpeed3pm Variable

Q1 = df1['WindSpeed3pm'].describe()[4]
Q2 = df1['WindSpeed3pm'].describe()[6]
IQR = Q2 - Q1
Lower_fence = Q1 - (IQR * 3)
Upper_fence = Q2 + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

df1['WindSpeed3pm'].describe()

''' For WindSpeed3pm, the minimum and maximum values are 0.0 and 87.0. So, the outliers are values > 57.0. '''

# 7. MULTIVARIATE DATA ANALYSIS

"""
      * An important step in EDA is to discover patterns and relationships between variables in the dataset.

      * We will use heat map and pair plot to discover the patterns and relationships in the dataset.

      * First of all, We will draw a heat map.
"""


correlation = df1.corr()

""" Heat  Map """

plt.figure(figsize = (16,12))
plt.title('Correlation HeatMap of Rain in Australia Dataset')
ax = sns.heatmap(correlation, square=True,annot=True,fmt='.2f',linecolor = 'white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
plt.show()

'''
       Interpretation :

       From the above correlation heat map, we can conclude that :-

           * MinTemp and MaxTemp variables are highly positively correlated (correlation coefficient = 0.74).

           * MinTemp and Temp3pm variables are also highly positively correlated (correlation coefficient = 0.71).

           * MinTemp and Temp9am variables are strongly positively correlated (correlation coefficient = 0.90).

           * MaxTemp and Temp9am variables are strongly positively correlated (correlation coefficient = 0.89).

           * MaxTemp and Temp3pm variables are also strongly positively correlated (correlation coefficient = 0.98).

           * WindGustSpeed and WindSpeed3pm variables are highly positively correlated (correlation coefficient = 0.69).
   
           * Pressure9am and Pressure3pm variables are strongly positively correlated (correlation coefficient = 0.96).

           * Temp9am and Temp3pm variables are strongly positively correlated (correlation coefficient = 0.86).

        Pair Plot :

First of all, I will define extract the variables which are highly positively correlated.

'''

num_var = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'WindGustSpeed', 'WindSpeed3pm', 'Pressure9am', 'Pressure3pm']

""" Now, we will draw pairplot to depict relationship between these variables."""

sns.pairplot(df1[num_var], kind='scatter', diag_kind='hist', palette='Rainbow')
plt.show()

"""    Interpretation :
    
          * I have defined a variable num_var which consists of MinTemp, MaxTemp, Temp9am, Temp3pm, WindGustSpeed, WindSpeed3pm, Pressure9am and Pressure3pm variables.

          * The above pair plot shows relationship between these variables.

"""

# 8. DECLARE FEATURE VECTOR AND TARGET VARIABLE

X = df1.drop(['RainTomorrow'],axis=1)

y = df1['RainTomorrow']



# 9. SPLIT DATA INTO SEPARATE TRAINING AND TEST SET

# Split X and y into training and testing sets

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#check the shape of X_train and X_test

X_train.shape, X_test.shape

# 10. FEATURE ENGINEERING

""" Feature Engineering is the process of transforming raw data into useful features that help us to understand our model better and increase its predictive power. We will carry out feature engineering on different types of variables.

   First, We will display the categorical and numerical variables again separately.

"""

#check data types in X_train

X_train.dtypes

#display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical

#display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical

""" Engineering missing values in numerical variables """

# check missing values in numerical variables in X_train

X_train[numerical].isnull().sum()

# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()

# print percentage of missing values in the numerical variables in training set

for col in numerical:
  if X_train[col].isnull().mean()>0:
    print(col, round(X_train[col].isnull().mean(),4))
    


* We assume that the data are missing completely at random (MCAR). There are two methods which can be used to impute missing values. One is mean or median imputation and other one is random sample imputation. When there are outliers in the dataset, we should use median imputation. So, we will use median imputation because median imputation is robust to outliers.

* We will impute missing values with the appropriate statistical measures of the data, in this case median. Imputation should be done over the training set, and then propagated to the test set. It means that the statistical measures to be used to fill missing values both in train and test set, should be extracted from the train set only. This is to avoid overfitting.
""
# impute missing values in X_train and X_test with respective column median in X_train

for df2 in [X_train, X_test]:
  for col in numerical:
    col_median = X_train[col].median()
    df2[col].fillna(col_median, inplace = True)

#check again missing values in numerical variables in X_train

X_train[numerical].isnull().sum()

#check missing values in  numerical variables in X_test

X_test[numerical].isnull().sum()

""" 
    Now, we can see that there are no missing values in the numerical columns of training and test set

    Engineering missing values in categorical variables
    
"""

#print percentage of missing values in the categorical variables in training set

X_train[categorical].isnull().mean()

#print categorical variables with missing data

for col in categorical:
  if X_train[col].isnull().mean()>0:
    print(col, (X_train[col].isnull().mean()))

#impute mising categorical variables with most frequent values

for df3 in [X_train, X_test]:
  df3['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0],inplace=True)
  df3['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0],inplace=True)
  df3['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0],inplace=True)
  df3['RainToday'].fillna(X_train['RainToday'].mode()[0],inplace=True)

df3.isnull().sum()

#now check the missing values

X_test[categorical].isnull().sum()

""" Finally checking the missing values in X_train & X_test"""

#checking the missing values in Test data

X_test.isnull().sum()

#checking the missing values in Train data

X_train.isnull().sum()

"""      Observation - There are no missing values in test and train data

Engineering outliers in numerical variables

As we have seen that the Rainfall, Evaporation,WindSpeed9am and WindSpeed3pm

"""

def max_value(df4,variable,top):
  return np.where(df4[variable]>top, top,df4[variable])

for df4 in [X_train, X_test]:
  df4['Rainfall'] = max_value(df4,'Rainfall',3.2)
  df4['Evaporation'] = max_value(df4,'Evaporation',21.8)
  df4['WindSpeed9am'] = max_value(df4,'WindSpeed9am',55)
  df4['WindSpeed3pm'] = max_value(df4,'WindSpeed3pm',57)

X_train.Rainfall.max(),X_test.Rainfall.max()

X_train.Evaporation.max(),X_test.Evaporation.max()

X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()

X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()

X_train[numerical].describe()

""" We can now see that the outliers in Rinafall, Evaporation, WindSpeed9am, and WindSpeed3pm columns are capped

    Encode categorical values**
"""

#print categorical values

categorical

X_train[categorical].head()

X_train['RainToday'] = X_train['RainToday'].map({'Yes':1,'No':0})

X_train.head()

X_train.dtypes

""" We can see two additional variables RainToday_0 and RainToday_1 are created from RainToday variable"""

X_train = pd.concat([X_train[numerical],X_train[['RainToday']], pd.get_dummies(X_train.Location,prefix='Location', prefix_sep='_'),pd.get_dummies(X_train.WindGustDir,prefix='WindGustDir', prefix_sep='_'),pd.get_dummies(X_train.WindDir9am,prefix='WindDir9am', prefix_sep='_'),pd.get_dummies(X_train.WindDir3pm,prefix='WindDir3pm', prefix_sep='_')], axis=1)

X_train.head()

# list(X_train.columns)

X_train.dtypes.nunique()

X_test['RainToday'] = X_test['RainToday'].map({'Yes':1,'No':0})

""" Similarly we will do it for X_test testing set"""

X_test = pd.concat([X_test[numerical], X_test[['RainToday']],
                     pd.get_dummies(X_test.Location,prefix='Location', prefix_sep='_'),
                     pd.get_dummies(X_test.WindGustDir,prefix='WindGustDir', prefix_sep='_'),
                     pd.get_dummies(X_test.WindDir9am,prefix='WindDir9am', prefix_sep='_'),
                     pd.get_dummies(X_test.WindDir3pm,prefix='WindDir3pm', prefix_sep='_')], axis=1)

print(X_test.shape)
print(X_test.shape)

""" 
     We now have training and testing set ready for model building. Before that, we should map all the feature variables onto the same scale. It is called feature scaling. I will do it as follows.

"""

# 11. FEATURE SCALING

X_train.describe()

cols = X_train.columns

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns = [cols])

X_test = pd.DataFrame(X_test, columns=[cols])

X_train.describe()

"""
    We now have X_train dataset ready to be fed into the Logistic Regression classifier. I will do it as follows.

"""

# 12. MODEL TRAINING

from sklearn.linear_model import LogisticRegression

#instantiate the model
logreg = LogisticRegression(solver='liblinear',random_state=0)

#fit the model
logreg.fit(X_train, y_train)

# save the model to disk
filename = 'logreg.pkl'
pickle.dump(logreg, open(filename, 'wb'))

# 13. PREDICT RESULT

y_pred_test = logreg.predict(X_test)

y_pred_test

"""predict_proba method

* **predict_proba** method gives the probabilities for the target variable(0 and 1) in this case, in array form.

* 0 is for probability of no rain and 1 is for probability of rain
"""

# probability of getting output as 0 - no rain

logreg.predict_proba(X_test)[:,0]

#probability of getting output as 1 - rain

logreg.predict_proba(X_test)[:,1]

# 14. CHECK ACCURACY SCORE

from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

"""
     As we can see y_test in the known values and y_pred_test are the predicted values so checked the accuracy which comes to be 85%

     Compare the train-set and test-set accuracy
     
"""

y_pred_train = logreg.predict(X_train)

y_pred_train

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

""" Check for overfitting and underfitting """

print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))

"""
    In Logistic Regression, we use default value of C = 1. It provides good performance with approximately 85% accuracy on both the training and the test set. But the model performance on both the training and test set are very comparable. It is likely the case of underfitting.

    We will increase C and fit a more flexible model.
"""

#fit the Logistic Regression model with C=100

#instantiate the model
logreg100 = LogisticRegression(C=100,solver='liblinear',random_state=0)

#fit the model
logreg100.fit(X_train,y_train)

# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))

""" 
    We can see that, C=100 results in higher test set accuracy and also a slightly increased training set accuracy. So, we can conclude that a more complex model should perform better.

    Now, I will investigate, what happens if we use more regularized model than the default value of C=1, by setting C=0.01.
"""

# fit the Logsitic Regression model with C=001

# instantiate the model
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)


# fit the model
logreg001.fit(X_train, y_train)

# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))

"""
    So, if we use more regularized model by setting C=0.01, then both the training and test set accuracy decrease relative to the default parameters.

    Compare model accuracy with null accuracy
    So, the model accuracy is 0.8501. But, we cannot say that our model is very good based on the above accuracy. We must compare it with the null accuracy. Null accuracy is the accuracy that could be achieved by always predicting the most frequent class.

    So, we should first check the class distribution in the test set.
    
"""

#check the class distribution in test set

y_test.value_counts()

"""We can see that the occurences of most frequent class is 22726. So, we can calculate null accuracy by dividing 22726 by total number of occurences."""

#check the null accuracy score

null_accuracy = ((22726)/(22726+6366))

print('Null accuracy score: {0:0.4f}'.format(null_accuracy))

"""
       Interpretation :
           
          * We can see that our model accuracy score is **0.8488** but null accuracy score is 0.7812.
          * So, we can conclude that our Logistic Regression model is doing a very good job in predicting the class labels.
          * But, it does not give the underlying distribution of values. Also, it does not tell anything about the type of errors our classifer is making.
          * We have another tool called Confusion matrix that comes to our rescue

"""
# 15. CONFUSION MATRIX

# print the confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n',cm)

print('\nTrue Positives(TP) = ',cm[0,0])

print('\nTrue Negatives(TP) = ',cm[1,1])

print('\nFalse Positives(TP) = ',cm[0,1])

print('\nFalse Negatives(TP) = ',cm[1,0])

"""
     The confusion matrix shows 21543 + 3139 = 24177 correct predictions and 3227 + 1183 = 4262 incorrect predictions.

In this case, we have

* True Positives (Actual Positive:1 and Predict Positive:1) - 21543
* True Negatives (Actual Negative:0 and Predict Negative:0) - 3139
* False Positives (Actual Negative:0 but Predict Positive:1) - 1183 (Type I error)
* False Negatives (Actual Positive:1 but Predict Negative:0) - 3227 (Type II error)

"""

# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

# 16. CLASSIFICATION METRICS

#Classificaton Report**

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))

"""  Classification Accuracy  """

TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

#print classificaiton accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

""" Classification Error """

# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))

"""**Comparing Different algorithms and their accuracy**

***
**1. Logistic Regression**
***
"""

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
predictions = lr.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

"""***
**2. Decision Trees**
***
"""

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
predictions = dt.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

"""***
**3. Random Forest**
***
"""

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
predictions = rf.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

"""**Hence we conclude that on comparing the there Alogithms Logistic Regression proves to be most accurate**"""

# 17. MODEL EVALUATION AND IMPROVEMENT

"""In this section, We will employ k-fold cross validation technique to improve the model performance.

***
**k-Fold Cross Validation**
***
"""

# Applying 5-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))

"""We can summarize the cross-validation accuracy by calculating its mean"""

# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))

"""Our, original model score is found to be 0.8488. The average cross-validation score is 0.8481. So, we can conclude that cross-validation does not result in performance improvement."""

# Making pickle file for our model

pickle.dump(lr, open("model.pkl", "wb"))

"""# **18. Results and Conclusion**

1. The logistic regression model accuracy score is 0.8488. So, the model does a very good job in predicting whether or not it will rain tomorrow in Australia.

2. Small number of observations predict that there will be rain tomorrow. Majority of observations predict that there will be no rain tomorrow.

3. The model shows no signs of overfitting.

4. Increasing the value of C results in higher test set accuracy and also a slightly increased training set accuracy. So, we can conclude that a more complex model should perform better.

5. Our, original model score is found to be 0.8488. The average cross-validation score is 0.8481. So, we can conclude that cross-validation does not result in performance improvement.
"""