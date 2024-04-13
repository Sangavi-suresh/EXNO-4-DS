# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/09cec392-423e-4d11-ac94-d9ff0a36a502)


```

data.isnull().sum()
```

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/dd0b24a8-db5f-4d35-80cf-bd22e12c8aca)

```

missing=data[data.isnull().any(axis=1)]
missing
```

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/f97d32d8-f6f4-4f42-9c78-5555cd8a9803)


```

data2=data.dropna(axis=0)
data2
```

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/72bb8410-6b47-413c-ad3f-17ef945e24b0)


```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/6b7a3a68-c985-4490-a956-472fdb4d59f2)


```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/1e8f8f7e-a502-4235-b6da-6924f3a8cb8e)


```

data2

```

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/94718a8e-302d-4b80-9365-3dd58768b00a)


```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/a24cc11f-e8d9-428a-819b-ad81082402ee)


```

columns_list=list(new_data.columns)
print(columns_list)

```

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/b2c13343-924d-456b-b671-57e759fbbe3e)


```


features=list(set(columns_list)-set(['SalStat']))
print(features)
```

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/28c1db26-10bf-4fd4-8fa3-6e4263d267ed)


```
y=new_data['SalStat'].values
print(y)

```

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/0acbdf6b-edc4-41ff-bceb-18c3e5d8dade)


```

x=new_data[features].values
print(x)

```

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/28ef6da4-d912-4506-882f-b6296186f04c)


```

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```


![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/bf06e4d1-df44-4bb1-ba86-d980318427cc)


```

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/78d3191b-60a0-41bd-ac5d-67ab34903624)




accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)


![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/1e49508b-dcf6-4117-81db-63429621690c)


print("Misclassified Samples : %d" % (test_y !=prediction).sum())

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/daaef5b9-7473-4547-91c7-6b8d95b48156)


data.shape

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/59090e8d-a449-48ed-9ce3-1c32358c7096)


```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

```

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/1eda0745-57f9-4efd-9b7c-fd4cf1dd052d)


```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/29848070-de88-41fc-90fa-21f5cbd81a32)



tips.time.unique()

![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/7e5c62e7-18b7-4194-963e-5e1808815d72)



contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)


![image](https://github.com/Sangavi-suresh/EXNO-4-DS/assets/118541861/61e3bebe-1ff1-4991-b4d1-2e462d4916b2)




















# RESULT:
       # INCLUDE YOUR RESULT HERE
