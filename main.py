import tensorflow as tf
from pandas.plotting._matplotlib import scatter_matrix
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyodbc
import os

# database = '[SK-BB]'
database = '[CZ-PREOS_GTN]'
zastavky = ['Ostrava-Svinov', 'Polanka n. O.', 'Jistebník', 'Studénka', 'Suchdol nad Odr.', 'Polom',
                'Hranice na Mor.', 'Drahotue', 'Lipník nad Beèv.', 'Prosenice', 'Pøerov os.n.']

# https://en.wikipedia.org/wiki/Interquartile_range#Outliers
# pre dany stlpec vypocita hranice


def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1, Q3 = np.percentile(datacolumn, [25, 75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range


def track_monitor(stops_list, df_p, title, directory):
    cesty = {}
    lastId = -1
    for index, row in df_p.iterrows():
        trainId = row['TrainId']
        if (row['FromName'] in stops_list) & (row['ToName'] in stops_list):
            if trainId == lastId:
                cesty.setdefault(trainId, []).append(row['DelayDiffPercent'])
            elif (row['SectIdx'] == 0) & (row['FromName'] == stops_list[0]):
                cesty[row['TrainId']] = [0.0, row['DelayDiffPercent']]
                lastId = trainId
        else:
            if trainId == lastId:
                cesty.pop(lastId, None)

    total = [0] * len(stops_list)

    trainsCount = 0
    for value in cesty.values():
        if len(value) == 11:
            plt.plot(zastavky, value)
            total = np.add(value, total)
            trainsCount += 1
    plt.title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='x-small')
    plt.savefig(directory + 'AllTrains')
    plt.close()

    dividers = [trainsCount] * len(stops_list)

    means = np.divide(total, dividers)

    plt.title(title)
    plt.plot(zastavky, means)
    plt.savefig(directory + 'Mean')
    plt.close()


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

cnxn = pyodbc.connect("Driver={SQL Server};"
                      "Server=dokelu.kst.fri.uniza.sk;"
                      "Database=TrainsDB20-01-23;"
                      "uid=Lukas;pwd=lukas")

df = pd.read_sql_query(
      'SELECT TOP(100000) [TrainId]'
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
      ',SectIdx'
      ',CASE WHEN DATEPART(MONTH,DepPlanTime) in (3,4,5) then \'Jar\''
	        ' WHEN DATEPART(MONTH,DepPlanTime) in (6,7,8) then \'Leto\''
			' WHEN DATEPART(MONTH,DepPlanTime) in (9,10,11) then \'Jesen\''
 			' WHEN DATEPART(MONTH,DepPlanTime) in (12,1,2) then \'Zima\''
			' ELSE \'ErrorValue\''
	  ' END as Season'
      ',CASE WHEN DATEPART(HOUR,DepRealTime) in (5,6,7,8,9,10) THEN \'Rano\''
			' WHEN DATEPART(HOUR,DepRealTime) in (11,12,13,14,15,16) THEN \'Obed\''
			' WHEN DATEPART(HOUR,DepRealTime) in (17,18,19,20,21,22) THEN \'Vecer\''
			' WHEN DATEPART(HOUR,DepRealTime) in (23,0,1,2,3,4) THEN \'Noc\''
	  ' END as DayTime'
    ' FROM [TrainsDb20-01-23].[dbo].' + database +
    ' where DepPlanTime IS NOT NULL'
    ' and DepRealTime IS NOT NULL'
    ' and ArrRealTime IS NOT NULL'
    ' and ArrPlanTime IS NOT NULL'
    ' and Weight > 0'
    ' and Length > 0'
    ' and CarCount > 0'
    ' and AxisCount > 0'
    ' and LengthSect > 0'
    ' and TrainType not in (\'Lv\', \'Sluz\')'
    ' and FromName <> ToName'
    ' order by TrainId,SectIdx ASC'
    , cnxn)

df['PredDelay'] = 0
df['PredLength'] = 0
df['DelayDiff'] = 0
df['DelayDiffPercent'] = 0.0
df['NoStop'] = 1


# vsetky typy vlakov v databaze
train_types = ['Ex', 'Lv', 'Mn', 'Nex', 'Os', 'PMD', 'Pn', 'R', 'Sluz', 'Sp', 'Sv', 'Vlec']
stop_types = ['ZaciatokTrasy', 'PokracovanieTrasy']
season_types = ['Jar', 'Leto', 'Jesen', 'Zima']
daytime_types = ['Rano', 'Obed', 'Vecer', 'Noc']

seasons_dataframes = {}
stops_dataframes = {}
train_type_dataframes = {}
daytimes_dataframes = {}

filters = {'TrainTypes': [train_types, train_type_dataframes, 'TrainType'],
           'Seasons': [season_types, seasons_dataframes, 'Season'],
           'DayTimes': [daytime_types, daytimes_dataframes, 'DayTime']}

print(df.shape)

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


lastId = -1
lastKm = 0
lastDelay = 0

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

df = df[(df['AxisCount'] / df['CarCount'] == 2) | (df['AxisCount'] / df['CarCount'] == 4)]
print(df.shape)

# ostranenie zapornych trvani tras
# df = df[(df['PlanDrivingTime'] > 60)]
# df = df[(df['RealDrivingTime'] > 60)]


input_attributes = ['Weight', 'Length', 'CarCount', 'AxisCount', 'PlanDrivingTime', 'LengthSect', 'PredLength', 'PredDelay']
cut_attributes = ['Weight', 'Length', 'CarCount', 'AxisCount', 'PlanDrivingTime', 'LengthSect', 'PredLength', 'PredDelay', 'DelayDiffPercent']

df['DelayDiff'] = (df['RealDrivingTime'] - df['PlanDrivingTime'])
# zosekava
# df = df[((df['DelayDiff'] < 43200) & (df['DelayDiff'] > - 43200))]

df['DelayDiffPercent'] = (df['DelayDiff'] / df['PlanDrivingTime'])


# print(df.describe)
for stopCountType in stop_types:
    if stopCountType == 'Zaciatok':
        stopCount = df[(df['SectIdx'] == 0)]
    else:
        stopCount = df[(df['SectIdx'] != 0)]
    print(stopCountType)
    print(stopCount.describe())

    if not os.path.exists(database + '/StopCount/' + stopCountType):
        os.makedirs(database + '/StopCount/' + stopCountType)
        os.makedirs(database + '/StopCount/' + stopCountType + '/ScatterPlots')

    if len(stopCount.index) > 30:
        cor = stopCount[cut_attributes].corr()
        print(cor)
        print(cor["DelayDiffPercent"].sort_values(ascending=False))

        # korelacna matica s hodnotami
        fig, ax = plt.subplots()
        ax.matshow(cor, cmap='seismic')
        for (i, j), z in np.ndenumerate(cor):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        plt.title(stopCountType)
        # plt.show()
        plt.savefig(database + '/StopCount/' + stopCountType + '/correlation')
        plt.close()

        # histogram hodnot rocneho obdobia
        stopCount[cut_attributes].hist(bins=30, figsize=(20, 15))
        plt.suptitle(stopCountType)
        # plt.show()
        plt.savefig(database + '/StopCount/' + stopCountType + '/histogram')
        plt.close()

        for attr in input_attributes:
            # scatterPlot plyvu atributu na hladanu premennu
            stopCount[cut_attributes].plot(kind='scatter', x=attr, y='DelayDiffPercent', alpha='0.3')
            plt.title(stopCountType)
            # plt.show()
            plt.savefig(database + '/StopCount/' + stopCountType + '/ScatterPlots/' + attr)
            plt.close()

        stops_dataframes[stopCountType] = stopCount


for filter_type, filter in filters.items():
    data_types = filter[0]
    dataframe = filter[1]
    column = filter[2]

    for data_type in data_types:
        type = df[(df[column] == data_type)]
        print(data_type)
        print(type.describe())

        if not os.path.exists(database + '/' + filter_type + '/' + data_type):
            os.makedirs(database + '/' + filter_type + '/' + data_type)
            os.makedirs(database + '/' + filter_type + '/' + data_type + '/ScatterPlots')
            os.makedirs(database + '/' + filter_type + '/' + data_type + '/TrainRoute')

        if len(type.index) > 30:
            cor = type[cut_attributes].corr()
            print(cor)
            print(cor["DelayDiffPercent"].sort_values(ascending=False))

            # korelacna matica s hodnotami
            fig, ax = plt.subplots()
            ax.matshow(cor, cmap='seismic')
            for (i, j), z in np.ndenumerate(cor):
                ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
            plt.title(data_type)
            # plt.show()
            plt.savefig(database + '/' + filter_type + '/' + data_type + '/correlation')
            plt.close()

            # histogram hodnot rocneho obdobia
            type[cut_attributes].hist(bins=30, figsize=(20, 15))
            plt.suptitle(data_type)
            # plt.show()
            plt.savefig(database + '/' + filter_type + '/' + data_type + '/histogram')
            plt.close()

            for attr in input_attributes:
                # scatterPlot plyvu atributu na hladanu premennu
                type[cut_attributes].plot(kind='scatter', x=attr, y='DelayDiffPercent', alpha='0.3')
                plt.title(data_type)
                # plt.show()
                plt.savefig(database + '/' + filter_type + '/' + data_type + '/ScatterPlots/' + attr)
                plt.close()

            dataframe[data_type] = type
            directory = database + '/' + filter_type + '/' + data_type + '/TrainRoute/'
            track_monitor(zastavky, type, data_type, directory)


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
if not os.path.exists(database + '/AllData'):
    os.makedirs(database + '/AllData')
    os.makedirs(database + '/AllData/ScatterPlots')

dffcut= df[cut_attributes]
dfflabels = df[['DelayDiffPercent']]

train_dataset = dffcut.sample(frac=0.8, random_state=0)
test_dataset = dffcut.drop(train_dataset.index)
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
print(train_stats.to_string())


cor = dffcut.corr()
print(cor["DelayDiffPercent"].sort_values(ascending=False))

fig, ax = plt.subplots()
ax.matshow(cor, cmap='seismic')

for (i, j), z in np.ndenumerate(cor):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

#plt.matshow(cor)

# plt.show()
plt.title('All Data')
plt.savefig(database + '/AllData/correlation')
plt.close()

dffcut.describe()
dffcut.hist(bins=50, figsize=(20, 15))
plt.suptitle("All data")
# plt.show()
plt.savefig(database + '/AllData/histogram')
plt.close()

for attr in input_attributes:
    dffcut.plot(kind='scatter', x=attr, y='DelayDiffPercent', alpha='0.3')
    plt.title("All_data")
    # plt.show()
    plt.savefig(database + '/AllData/ScatterPlots/' + attr)
    plt.close()
