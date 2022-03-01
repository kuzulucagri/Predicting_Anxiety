## Import Libaries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sbn
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,classification_report
from scipy.stats import pearsonr,spearmanr,kendalltau
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv("dataset.csv")
pd.options.display.max_columns = None

print(dataset)

print(dataset.dtypes.value_counts()) 
#In our dataset has 1 object value which is major

print("Dataset Describe\n",dataset.describe()) 
#This is the table for metrics in features.
#As we see in this table, age has outliers so we can remove it in the next steps.

print("Dataset CORR \n",dataset.corr()) 
#This table is for relationship between features.

print("Dataset SUM \n",dataset.sum())

print("Dataset MEAN \n",dataset.mean())

print("Dataset STD \n",dataset.std())

print("Dataset MIN \n",dataset.min())

print("Dataset MAX \n",dataset.max())

## Outliers and Cleaning

# Age
le = LabelEncoder()

le.fit(dataset.age)

print(list(le.classes_))

# In our **age** datas has outliers like 5555 and 33769 so we can declare 0 instead of them.
for i in range(len(dataset)):
    if dataset.age[i] > 75:
        dataset.iloc[i:i+1,63:64] = 0
        
# Gender "What is your gender?", 1=Male, 2=Female, 3=Other -> 0 variables have to change with 3
print(dataset.gender.value_counts())

for i in range(len(dataset)):
    if dataset.gender[i] == 0:
        dataset.iloc[i:i+1,61::62] = 3
        
# Urban "What type of area did you live when you were a child?", 1=Rural (country side), 2=Suburban, 3=Urban (town, city)
# 0 variables have to change with 3
print(dataset.urban.value_counts())

for i in range(len(dataset)):
    if dataset.urban[i] == 0:
        dataset.iloc[i:i+1,60:61] = 3
        
# Education "How much education have you completed?", 1=Less than high school, 2=High school, 3=University degree, 4=Graduate degree
# 0 variables have to change with 1
print(dataset.education.value_counts())

for i in range(len(dataset)):
    if dataset.education[i] == 0:
        dataset.iloc[i:i+1,59:60] = 1
        
# Hand "What hand do you use to write with?", 1=Right, 2=Left, 3=Both
# 0 variables have to change with 1
print(dataset.hand.value_counts())

for i in range(len(dataset)):
    if dataset.hand[i] == 0:
        dataset.iloc[i:i+1,64:65] = 1
        
# Religion "What is your religion?", 1=Agnostic, 2=Atheist, 3=Buddhist, 4=Christian (Catholic), 5=Christian (Mormon), 6=Christian (Protestant), 7=Christian (Other), 8=Hindu, 9=Jewish, 10=Muslim, 11=Sikh, 12=Other
# 0 variables have to change with 12
print(dataset.religion.value_counts())

for i in range(len(dataset)):
    if dataset.religion[i] == 0:
        dataset.iloc[i:i+1,65:66] = 12

# Race "What is your race?", 1=Asian, 2=Arab, 3=Black, 4=Indigenous Australian, Native American or White***, 5=Other
# 0 variables have to change with 5
print(dataset.race.value_counts())

for i in range(len(dataset)):
    if dataset.race[i] == 0:
        dataset.iloc[i:i+1,67:68] = 5
        
# Voted "Have you voted in a national election in the past year?", 1=Yes, 2=No
# 0 variables have to change with 2
dataset.voted.value_counts()

for i in range(len(dataset)):
    if dataset.voted[i] == 0:
        dataset.iloc[i:i+1,68:69] = 2
        
        
# Married "What is your marital status?", 1=Never married, 2=Currently married, 3=Previously married
# 0 variables are outliers so we can change it with 1
print(dataset.married.value_counts())

for i in range(len(dataset)):
    if dataset.married[i] == 0:
        dataset.iloc[i:i+1,69:70] = 1
        
# Family Size
print(dataset.familysize.value_counts())

print(dataset.major.value_counts()) 
#We have 580 type of major so we should not convert it to numeric value.


## Missing Values

print(dataset.isnull().sum()) 
# Except major, we have no missing values so we can fill in major with 0

dataset.major.fillna(0)

## Histograms


plt.hist(dataset["age"],bins = [10,15,20,25,30,35,40,45,50,55,60,65,70,75]) 
#As we know that our age range is like 10 to 75 this is why we can declare bins like that.
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Value")
plt.show()

plt.hist(dataset["religion"],bins = 12) 
# We have 12 types of religion.
plt.title("Religion Distribution")
plt.xlabel("Religion")
plt.ylabel("Value")
plt.show()

plt.hist(dataset["race"],bins = 5) 
# Our race feature has 5 types.
plt.title("Race Distribution")
plt.xlabel("Race")
plt.ylabel("Value")
plt.show()

plt.hist(dataset["voted"],bins = 5)
plt.title("Voted Distribution")
plt.xlabel("Vote-NonVoted")
plt.ylabel("Value")
plt.show()

plt.hist(dataset["familysize"],bins = [0,1,2,3,4,5,6,7,8,9,10])
plt.title("Family Size Distribution")
plt.xlabel("Family Size")
plt.ylabel("Value")
plt.show()

'''
In the next step, we can use histogram for analyze the TIPI values

TIPI1 Extraverted, enthusiastic.

TIPI2 Critical, quarrelsome.

TIPI3 Dependable, self-disciplined.

TIPI4 Anxious, easily upset.

TIPI5 Open to new experiences, complex.

TIPI6 Reserved, quiet.

TIPI7 Sympathetic, warm.

TIPI8 Disorganized, careless.

TIPI9 Calm, emotionally stable.

TIPI10 Conventional, uncreative.
'''

kwargs = dict(alpha=0.5, bins=15)

plt.hist(dataset.TIPI1, **kwargs, color='y', label='TIPI1')
plt.hist(dataset.TIPI2, **kwargs, color='b', label='TIPI2')
plt.gca().set(title='Histogram of Dataset', ylabel='Values')
plt.xlim(0,7)
plt.legend();

kwargs = dict(alpha=0.5, bins=15)

plt.hist(dataset.TIPI3, **kwargs, color='y', label='TIPI3')
plt.hist(dataset.TIPI4, **kwargs, color='b', label='TIPI4')
plt.gca().set(title='Histogram of Dataset', ylabel='Values')
plt.xlim(0,7)
plt.legend();

kwargs = dict(alpha=0.5, bins=15)

plt.hist(dataset.TIPI5, **kwargs, color='y', label='TIPI5')
plt.hist(dataset.TIPI6, **kwargs, color='b', label='TIPI6')
plt.gca().set(title='Histogram of Dataset', ylabel='Values')
plt.xlim(0,7)
plt.legend();

kwargs = dict(alpha=0.5, bins=15)

plt.hist(dataset.TIPI7, **kwargs, color='y', label='TIPI7')
plt.hist(dataset.TIPI8, **kwargs, color='b', label='TIPI8')
plt.gca().set(title='Histogram of Dataset', ylabel='Values')
plt.xlim(0,7)
plt.legend();

kwargs = dict(alpha=0.5, bins=15)

plt.hist(dataset.TIPI9, **kwargs, color='y', label='TIPI9')
plt.hist(dataset.TIPI10, **kwargs, color='b', label='TIPI10')
plt.gca().set(title='Histogram of Dataset', ylabel='Values')
plt.xlim(0,7)
plt.legend();

print(dataset.corr()["education"].sort_values()) 
#In here we are checking the correlation education with the others and sort then we can use box blot with the lowest correlation features

## Box Plot

sbn.catplot(x = "education", y = "voted",kind = "box",data = dataset) 
#Also we can use box plot for analyze relationship between features

print(dataset.corr()["gender"].sort_values())

sbn.catplot(x = "gender", y = "TIPI9",kind = "box",data = dataset)

print(dataset.corr()["age"].sort_values())

sbn.catplot(x = "voted", y = "age",kind = "box",data = dataset)

print(dataset.corr()["hand"].sort_values())

## Bar Plot

sbn.catplot(x = "hand", y = "VCL4",kind = "bar",data = dataset) 
#Bar plot is also using for visualization as you see.

print(dataset.corr()["married"].sort_values())

sbn.catplot(x = "married", y = "TIPI8",kind = "bar",data = dataset)

print(dataset.corr()["familysize"].sort_values())

sbn.catplot(x = "urban", y = "familysize",kind = "bar",data = dataset)


from sklearn.model_selection import train_test_split


##Preprocessing phase has done...

X = dataset.drop(labels = ["gender","major"],axis = 1)
#In my dataset I want use age value as target so I have to split my values X and y.
#X has to be all numerical values except age so I can use drop function for remove gender and major.
y = dataset.gender

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42) 
# I used train test split function in here, and assign test_size as %30

# ALGORITHMS


# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss = "hinge",shuffle=True,random_state=42)

sgd.fit(X_train,y_train)

y_predict_sgd = sgd.predict(X_test)

# Metric Results for Stochastic Gradient Descent 
print("Stochastic Gradient Descent Model Score: ",sgd.score(X,y))
print("Stochastic Gradient Descent Mean Squared Error: ",mean_squared_error(y_test,y_predict_sgd))
print("Stochastic Gradient Descent Mean Absolute Error: ",mean_absolute_error(y_test,y_predict_sgd))
print("Stochastic Gradient Descent r2 Score: ",r2_score(y_test,y_predict_sgd))
print("Classification Report:\n ",classification_report(y_test,y_predict_sgd,digits = 2))


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="gini")

dtc.fit(X_train,y_train)

y_predict_dtc = dtc.predict(X_test)

# Metric Results for Decision Tree Classifier
print("Decision Classifier Model Score: ",dtc.score(X,y))
print("Decision Classifier Mean Squared Error: ",mean_squared_error(y_test,y_predict_dtc))
print("Decision Classifier Mean Absolute Error: ",mean_absolute_error(y_test,y_predict_dtc))
print("Decision Tree Classifier r2 Score: ",r2_score(y_test,y_predict_dtc))
print("Classification Report:\n ",classification_report(y_test,y_predict_dtc,digits = 2))

# K - Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3,metric="euclidean")

knn.fit(X_train,y_train)

y_predict_knn = knn.predict(X_test)

# Metric Results for K-Nearest Neighbors
print("K-Nearest Neighbors Model Score: ",knn.score(X,y))
print("K-Nearest Neighbors Mean Squared Error: ",mean_squared_error(y_test,y_predict_knn))
print("K-Nearest Neighbors Mean Absolute Error: ",mean_absolute_error(y_test,y_predict_knn))
print("K-Nearest Neighbors r2 Score: ",r2_score(y_test,y_predict_knn))
print("Classification Report:\n ",classification_report(y_test,y_predict_knn,digits = 2))

# Ensemble - Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100,criterion="gini")

rfc.fit(X_train,y_train)

y_predict_rfc = rfc.predict(X_test)

# Metric Results for Random Forest Regressor
print("Random Forest Regression Model Score: ",rfc.score(X,y))
print("Random Forest Regression Mean Squared Error: ",mean_squared_error(y_test,y_predict_rfc))
print("Random Forest Regression Mean Absolute Error: ",mean_absolute_error(y_test,y_predict_rfc))
print("Random Forest Regression r2 Score: ",r2_score(y_test,y_predict_rfc))
print("Classification Report:\n ",classification_report(y_test,y_predict_rfc,digits = 2))

#The best model score is in the Random Forest Regressor so we can use this algorithm

## Information Gain & Gini Index

def compute_impurity(feature, impurity_criterion):
    probs = feature.value_counts(normalize=True)
    
    if impurity_criterion == "entropy":
        impurity = -1 * np.sum(np.log2(probs) * probs)
    elif impurity_criterion == "gini":
        impurity = 1 - np.sum(np.square(probs))
    else:
        raise ValueError("Unknown impurity criterion")
        
    return(round(impurity, 3))

print("Gini Index: ",compute_impurity(dataset["gender"],"gini"))

target_entropy = compute_impurity(dataset["gender"], "entropy")
target_gini = compute_impurity(dataset["gender"],"gini")
information_gain = target_entropy - target_gini
print("Information Gain: ",information_gain)

## Similarity Metric Result

corr,_ = pearsonr(y_test,y_predict_rfc)
print("Pearson's Correlation: ",corr)

corr, _ = spearmanr(y_test,y_predict_rfc)
print("Spearman's Correlation: ",corr)

corr, _ = kendalltau(y_test, y_predict_rfc)
print("Kendall's Tau: ",corr)













