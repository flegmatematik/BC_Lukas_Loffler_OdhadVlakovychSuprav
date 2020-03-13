import tensorflow as tf
from pandas.plotting._matplotlib import scatter_matrix
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyodbc


def outlier_treatment(datacolumn, lower, upper):
    sorted(datacolumn)
    Q1, Q3 = np.percentile(datacolumn, [lower, upper])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range


cnxn = pyodbc.connect("Driver={SQL Server};"
                      "Server=dokelu.kst.fri.uniza.sk;"
                      "Database=TrainsDB20-01-23;"
                      "uid=Lukas;pwd=lukas")


df = pd.read_sql_query(
    'SELECT [Id]'
      ',[TrainId]'
      ',[FromName]'
      ',[ToName]'
      ',[TrainType]'
      ',[Weight]'
      ',[Length]'
      ',[CarCount]'
      ',[AxisCount]'
      ',SUBSTRING([EngineType],1,4) as EngineType'
      ',[ArrPlanTime]'
	  ',[DepPlanTime]'
      ',[ArrRealTime]'
      ',[DepRealTime]'
      ',DATEDIFF(SECOND,\'1970-01-01\',[DepRealTime]) as DepRealTimeStamp'
      ',DATEDIFF(SECOND,\'1970-01-01\',[ArrRealTime]) as ArrRealTimeStamp'
      ',COALESCE(DATEDIFF(SECOND,\'1970-01-01\',[ArrRealTime])-DATEDIFF(SECOND,\'1970-01-01\',[DepRealTime]),0) as RealDrivingTime'
      ',DATEDIFF(SECOND,\'1970-01-01\',[DepPlanTime]) as DepPlanTimeStamp'
      ',DATEDIFF(SECOND,\'1970-01-01\',[ArrPlanTime]) as ArrPlanTimeStamp'
      ',COALESCE(DATEDIFF(SECOND,\'1970-01-01\',[ArrPlanTime])-DATEDIFF(SECOND,\'1970-01-01\',[DepPlanTime]),0) as PlanDrivingTime'
      ',COALESCE(DATEDIFF(SECOND,\'1970-01-01\',[DepRealTime])-DATEDIFF(SECOND,\'1970-01-01\',[DepPlanTime]),0) as DelayDeparture'
      ',COALESCE(DATEDIFF(SECOND,\'1970-01-01\',[ArrRealTime])-DATEDIFF(SECOND,\'1970-01-01\',[ArrPlanTime]),0) as DelayArrive'
      ',COALESCE([LengthSect],0) as LengthSect'
      ',COALESCE([PredLength],0) as PredLength'
    ' FROM [TrainsDb20-01-23].[dbo].[SK-BB]'
    ' where DepPlanTime IS NOT NULL'
    ' and DepRealTime IS NOT NULL'
    ' and ArrRealTime IS NOT NULL'
    ' and ArrPlanTime IS NOT NULL'
    ' and Weight >= 5'
    ' and Length >= 5'
    ' and CarCount >= 2'
    ' and (AxisCount/CarCount = 4 or AxisCount/CarCount = 2)'
    ' and FromName <> ToName'
    ' order by TrainId,SectIdx ASC'
    , cnxn)


df['FromName'] = pd.Categorical(df['FromName'])
df['FromName'] = df.FromName.cat.codes
df['ToName'] = pd.Categorical(df['ToName'])
df['ToName'] = df.ToName.cat.codes
df['PredDelay'] = 0
df['DelayDiff'] = 0



lastId = -1
lastKm = 0
lastDelay = 0

train_types = ['Ex', 'Lv', 'Mn', 'Nex', 'Os', 'PMD', 'Pn', 'R', 'Sluz', 'Sp', 'Sv', 'Vlec']
train_type_dataframes = {}

print(df.shape)

for index in df.index:
    if df.at[index, 'TrainId'] == lastId:
        df.at[index, 'PredLength'] = lastKm
        df.at[index, 'PredDelay'] = lastDelay
    else:
        lastId = df.at[index, 'TrainId']

    lastKm = df.at[index, 'LengthSect']
    lastDelay = df.at[index, 'DelayArrive']

print(df.to_string)

# zaporne trvania tras
df = df[(df['PlanDrivingTime'] > 60)]
df = df[(df['RealDrivingTime'] > 60)]

# outliers
input_attributes = ['Weight', 'Length', 'CarCount', 'AxisCount', 'PlanDrivingTime', 'LengthSect', 'PredLength', 'PredDelay']
cut_attributes = ['Weight', 'Length', 'CarCount', 'AxisCount', 'PlanDrivingTime', 'LengthSect', 'PredLength', 'PredDelay', 'DelayDiff']

for attr in input_attributes:
    lowerbound, upperbound = outlier_treatment(df[attr], 10, 95)
    df.drop(df[(df[attr] > upperbound) | (df[attr] < lowerbound)].index, inplace=True)

df['DelayDiff'] = (df['DelayArrive'] - df['PredDelay'])
lowerbound, upperbound = outlier_treatment(df['DelayDiff'], 0.5, 99.5)
df.drop(df[(df['DelayDiff'] > upperbound) | (df['DelayDiff'] < lowerbound)].index, inplace=True)

for train_type in train_types:
    ttcut = df[(df['TrainType'] == train_type)]
    ttcut = ttcut[cut_attributes]
    train_type_dataframes[train_type] = ttcut

    print(train_type)
    print(train_type_dataframes[train_type].shape)

    if len(ttcut.index) > 1:
        cor = ttcut.corr()
        print(cor["DelayDiff"].sort_values(ascending=False))
        fig, ax = plt.subplots()
        ax.matshow(cor, cmap='seismic')
        for (i, j), z in np.ndenumerate(cor):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        plt.title(train_type)
        plt.show()
        ttcut.hist(bins=40, figsize=(20, 15))
        plt.title(train_type)
        plt.show()

        for attr in input_attributes:
            ttcut.plot(kind='scatter', x=attr, y='DelayDiff', alpha='0.3')
            plt.title(train_type)
            plt.show()

        """
        ttcut.plot(kind='scatter', x='Weight', y='DelayDiff', alpha='0.3')
        plt.title(train_type)
        plt.show()
        """

# print(df.to_string())
print(df.dtypes)

'''
df.isna().sum()
df = df.dropna()
'''

X = np.c_[df['AxisCount']]
Y = np.c_[df['DelayArrive']]
"""
df.plot.scatter(x='LengthSect', y='DelayArrive')
plt.show()
"""

dffcut= df[cut_attributes]
dfflabels = df[['DelayDiff']]

train_dataset = dffcut.sample(frac=0.8, random_state=0)
test_dataset = dffcut.drop(train_dataset.index)
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
print(train_stats.to_string())


cor = dffcut.corr()
print(cor["DelayDiff"].sort_values(ascending=False))

fig, ax = plt.subplots()
ax.matshow(cor, cmap='seismic')

for (i, j), z in np.ndenumerate(cor):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

#plt.matshow(cor)
plt.show()

dffcut.describe()
dffcut.hist(bins=50, figsize=(20, 15))
plt.title("All data")
plt.show()


for attr in input_attributes:
    dffcut.plot(kind='scatter', x=attr, y='DelayDiff', alpha='0.3')
    plt.title("All_data")
    plt.show()