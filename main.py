import tensorflow as tf
from pandas.plotting._matplotlib import scatter_matrix
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyodbc

# https://en.wikipedia.org/wiki/Interquartile_range#Outliers
# pre dany stlpec vypocita hranice
def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1, Q3 = np.percentile(datacolumn, [25, 75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

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
    ' and Weight > 0'
    ' and Length > 0'
    ' and CarCount > 0'
    ' and AxisCount > 0'
    ' and LengthSect > 0'
    ' and FromName <> ToName'
    ' order by TrainId,SectIdx ASC'
    , cnxn)


df['FromName'] = pd.Categorical(df['FromName'])
df['FromName'] = df.FromName.cat.codes
df['ToName'] = pd.Categorical(df['ToName'])
df['ToName'] = df.ToName.cat.codes
df['PredDelay'] = 0
df['PredLength'] = 0
df['DelayDiff'] = 0

lastId = -1
lastKm = 0
lastDelay = 0
# vsetky typy vlakov v databaze
train_types = ['Ex', 'Lv', 'Mn', 'Nex', 'Os', 'PMD', 'Pn', 'R', 'Sluz', 'Sp', 'Sv', 'Vlec']
train_type_dataframes = {}

print(df.shape)

# vymena nespravnych stlpcov

# df.loc[cond, ['AxisCount', 'CarCount']] = df.loc[cond, ['CarCount', 'AxisCount']]

# filtracia
print(df.shape)
# vsetky nemozne udaje - Pri zmene na letny a zimny cas sa nemozne hodnoty nevyskytuju
# df = df.drop(df[(df['PlanDrivingTime'] <= 60)].index, inplace=True)
df = df[(df['PlanDrivingTime'] > 60)]
print(df.shape)
# zmeny casu - NA ZIMNY

# -------------------------------2016-------------------------------------------------
df.loc[(((df['DepRealTimeStamp'] > 1477792800) & (df['DepRealTimeStamp'] < 1477807200))
        # ------------------------2017-------------------------------------------------
        | ((df['DepRealTimeStamp'] > 1509242400) & (df['DepRealTimeStamp'] < 1509256800))
        # ------------------------2018-------------------------------------------------
        | ((df['DepRealTimeStamp'] > 1540692000) & (df['DepRealTimeStamp'] < 1540706400))
        # ------------------------2019-------------------------------------------------
       | ((df['DepRealTimeStamp'] > 1572141600) & (df['DepRealTimeStamp'] < 1572156000)))
       & (df['DelayDeparture'] < -1500)
       & (df['DelayDeparture'] > -5000),
       ['DelayDeparture', 'DepRealTimeStamp']] += 3600
print(df.shape)
#  -------------------------------2016-------------------------------------------------
df.loc[(((df['ArrRealTimeStamp'] > 1477792800) & (df['ArrRealTimeStamp'] < 1477807200))
        # ------------------------2017-------------------------------------------------
        | ((df['ArrRealTimeStamp'] > 1509242400) & (df['ArrRealTimeStamp'] < 1509256800))
        # ------------------------2018-------------------------------------------------
        | ((df['ArrRealTimeStamp'] > 1540692000) & (df['ArrRealTimeStamp'] < 1540706400))
        # ------------------------2019-------------------------------------------------
       | ((df['ArrRealTimeStamp'] > 1572141600) & (df['ArrRealTimeStamp'] < 1572156000)))
       & (df['DelayArrive'] < -1500)
       & (df['DelayArrive'] > -5000),
       ['DelayArrive', 'ArrRealTimeStamp']] += 3600
print(df.shape)

# zmena casu ----------------- NA LETNY-----------------------------------------------
# -------------------------------2016-------------------------------------------------
df.loc[(((df['DepRealTimeStamp'] > 1459044000) & (df['DepRealTimeStamp'] < 1459058400))
        # ------------------------2017-------------------------------------------------
        | ((df['DepRealTimeStamp'] > 1490493600) & (df['DepRealTimeStamp'] < 1490508000))
        # ------------------------2018-------------------------------------------------
        | ((df['DepRealTimeStamp'] > 1521943200) & (df['DepRealTimeStamp'] < 1521957600))
        # ------------------------2019-------------------------------------------------
       | ((df['DepRealTimeStamp'] > 1553997600) & (df['DepRealTimeStamp'] < 1554012000)))
       & (df['DelayDeparture'] > 2200)
       & (df['DelayDeparture'] < 5700),
       ['DelayDeparture', 'DepRealTimeStamp']] -= 3600
print(df.shape)
#  -------------------------------2016-------------------------------------------------
df.loc[(((df['ArrRealTimeStamp'] > 1459044000) & (df['ArrRealTimeStamp'] < 1459058400))
        # ------------------------2017-------------------------------------------------
        | ((df['ArrRealTimeStamp'] > 1490493600) & (df['ArrRealTimeStamp'] < 1490508000))
        # ------------------------2018-------------------------------------------------
        | ((df['ArrRealTimeStamp'] > 1521943200) & (df['ArrRealTimeStamp'] < 1521957600))
        # ------------------------2019-------------------------------------------------
       | ((df['ArrRealTimeStamp'] > 1553997600) & (df['ArrRealTimeStamp'] < 1554012000)))
       & (df['DelayArrive'] > 2200)
       & (df['DelayArrive'] < 5700),
       ['DelayArrive', 'ArrRealTimeStamp']] -= 3600
print(df.shape)

# vypocet casu trvania cesty
df['RealDrivingTime'] = (df['ArrRealTimeStamp'] - df['DepRealTimeStamp'])
# vsetky udaje, kde prevoz trval viac ako 2 hodiny -> vacsinou nad 1 den = chyba
# df = df.drop(df[(df['PlanDrivingTime'] > 7200) & (df['RealDrivingTime'] > 7200)].index, inplace=True)
df = df[(df['PlanDrivingTime'] < 7200) | (df['RealDrivingTime'] < 7200)]
print(df.shape)

# vyhodnenie neplatnych trvani tras
df = df[(df['RealDrivingTime'] > 60)]


# print(df)

# vypocet hodnot pre dlzku predosleho useku a posledneho meskania
for index in df.index:
    # vymena stlpcov
    vymena = []
    vymena.insert(0, df.at[index, 'CarCount'])
    vymena.insert(1, df.at[index, 'AxisCount'])
    vymena.insert(2, df.at[index, 'Length'])
    vymena.sort()
    df.at[index, 'CarCount'] = vymena[0]
    df.at[index, 'AxisCount'] = vymena[1]
    df.at[index, 'Length'] = vymena[2]


    if df.at[index, 'TrainId'] == lastId:
        df.at[index, 'PredLength'] = lastKm
        df.at[index, 'PredDelay'] = lastDelay
    else:
        lastId = df.at[index, 'TrainId']

    lastKm = df.at[index, 'LengthSect']
    lastDelay = df.at[index, 'DelayArrive']

print(df.shape)

# ostranenie zapornych trvani tras
# df = df[(df['PlanDrivingTime'] > 60)]
# df = df[(df['RealDrivingTime'] > 60)]


input_attributes = ['Weight', 'Length', 'CarCount', 'AxisCount', 'PlanDrivingTime', 'LengthSect', 'PredLength', 'PredDelay']
cut_attributes = ['Weight', 'Length', 'CarCount', 'AxisCount', 'PlanDrivingTime', 'LengthSect', 'PredLength', 'PredDelay', 'DelayDiff']

df['DelayDiff'] = (df['DelayArrive'] - df['PredDelay'])
# zosekava
# df = df[((df['DelayDiff'] < 43200) & (df['DelayDiff'] > - 43200))]

for train_type in train_types:
    ttcut = df[(df['TrainType'] == train_type)]
    ttcut = ttcut[cut_attributes]

    print(train_type)
    print(ttcut.describe())

    if len(ttcut.index) > 30:
        cor = ttcut.corr()
        print(cor)
        print(cor["DelayDiff"].sort_values(ascending=False))

        # corelacna matica s hodnotami
        fig, ax = plt.subplots()
        ax.matshow(cor, cmap='seismic')
        for (i, j), z in np.ndenumerate(cor):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        plt.title(train_type)
        # plt.show()
        plt.savefig('BB/' + train_type + '/correlation')

        ttcut.hist(bins=30, figsize=(20, 15))
        plt.suptitle(train_type)
        # plt.show()
        plt.savefig('BB/' + train_type + '/histogram')

        for attr in input_attributes:
            '''
            # outliers - vyradenie pomocou IQR - Inter Quantile Range
            lowerbound, upperbound = outlier_treatment(ttcut[attr])
            ttcut.drop(ttcut[(ttcut[attr] > upperbound) | (ttcut[attr] < lowerbound)].index, inplace=True)
            '''
            # scatterPlot plyvu atributu na hladanu premennu
            ttcut.plot(kind='scatter', x=attr, y='DelayDiff', alpha='0.3')
            plt.title(train_type)
            # plt.show()
            plt.savefig('BB/' + train_type + '/ScatterPlots/' + attr)



        """
        ttcut.plot(kind='scatter', x='Weight', y='DelayDiff', alpha='0.3')
        plt.title(train_type)
        plt.show()
        """

        train_type_dataframes[train_type] = ttcut

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

# plt.show()
plt.title('All Data')
plt.savefig('BB/AllData/correlation')

dffcut.describe()
dffcut.hist(bins=50, figsize=(20, 15))
plt.suptitle("All data")
# plt.show()
plt.savefig('BB/AllData/histogram')

for attr in input_attributes:
    dffcut.plot(kind='scatter', x=attr, y='DelayDiff', alpha='0.3')
    plt.title("All_data")
    # plt.show()
    plt.savefig('BB/AllData/ScatterPlots/' + attr)
