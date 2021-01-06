##### Import packages
# Basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Other packages
from random import sample

# To avoid warnings
import warnings
warnings.filterwarnings("ignore")





##### Functions

# Percentile function for agregate
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_





##### Import data
# Check the csv's path before running it

dfx = pd.read_csv('accs_x.csv', names=['id_','time_ref','value_acc_x'], header=0)
dfy = pd.read_csv('accs_y.csv', names=['id_','time_ref','value_acc_y'], header=0)
dfz = pd.read_csv('accs_z.csv', names=['id_','time_ref','value_acc_z'], header=0)
df_energy = pd.read_csv('energy.csv')





##### Quick check of each dataset

# x-axis accelerometry
print(f' accs_x '.center(50,'#'))
print(dfx.info())
print(dfx.describe())
print(' ')

# y-axis accelerometry
print(f' accs_y '.center(50,'#'))
print(dfy.info())
print(dfy.describe())
print(' ')

# z-axis accelerometry
print(f' accs_z '.center(50,'#'))
print(dfz.info())
print(dfz.describe())
print(' ')

# energy associated with accelerometry and heart rate per time interval
print(f' energy '.center(50,'#'))
print(df_energy.info())
print(df_energy.describe())
print(' ')





##### Format change to datetime on some energy columns

for col in ['date_Hr', 'startDate_energy', 'endDate_energy']:
    df_energy[col] = pd.to_datetime(df_energy[col])


# # Creation of major variables




##### I create the variable 'jerk' which is the derivative of the acceleration.
# To do it properly, I do an approximation using the variable as discrete and assuming that the intervals of time are constant
# The acceleration is a vector, and therefore its derivative also has vectorial character.
# Then, I calculate the partial derivatives with respect to each axis.

dfx["jerk_x"]= dfx["value_acc_x"] - dfx.groupby("id_")["value_acc_x"].shift(-1)
dfy["jerk_y"] = dfy["value_acc_y"] - dfy.groupby("id_")["value_acc_y"].shift(-1)
dfz["jerk_z"] = dfz["value_acc_z"] - dfz.groupby("id_")["value_acc_z"].shift(-1)

dfx.drop(dfx.loc[dfx.jerk_x != dfx.jerk_x].index, inplace=True)
dfy.drop(dfy.loc[dfy.jerk_y != dfy.jerk_y].index, inplace=True)
dfz.drop(dfz.loc[dfz.jerk_z != dfz.jerk_z].index, inplace=True)

dfx.head()





##### I put together the information from the three axes in order to get more information.
# I create the acceleration module and jerk module variables, which are scalar magnitudes with crucial information 
# about the activity and I assumed that are relevant to the model.

dfx.time_ref = dfx.time_ref.apply(lambda x: x[x.rfind("_")+1:])
dfy.time_ref = dfy.time_ref.apply(lambda x: x[x.rfind("_")+1:])
dfz.time_ref = dfz.time_ref.apply(lambda x: x[x.rfind("_")+1:])

df_acc = pd.merge(pd.merge(dfx,dfy,how="inner",on=['id_', 'time_ref']),dfz,how="inner",on=['id_', 'time_ref'])

df_acc["mod_acc"] = np.sqrt((df_acc.value_acc_x**2) + (df_acc.value_acc_y**2) + (df_acc.value_acc_z**2))
df_acc["mod_jerk"] = np.sqrt((df_acc.jerk_x**2) + (df_acc.jerk_y**2) + (df_acc.jerk_z**2))

df_acc.head()


# # Exploratory Data Analysis




##### Acceleration module and jerk module variables over time
# To plot it faster, I choose 15 random ids

random_id = df_acc.loc[sample(range(0,len(df_acc.id_)),15)].id_.values
data = df_acc[df_acc.id_.isin(random_id)]

for var in ['mod_acc','mod_jerk']:
    plt.figure(figsize=(15, 10))
    plot = sns.lineplot('time_ref', var, data = data, hue = 'id_', legend=False)
    plt.title(f'{var} over time')
    plot.set(xlabel = None, xticklabels = [])
    plot.tick_params(bottom = False)
    plt.show()





##### x/y/z-axis accelerometry over time
# To plot it faster, I choose 15 random ids

random_id = df_acc.loc[sample(range(0,len(df_acc.id_)),15)].id_.values
data = df_acc[df_acc.id_.isin(random_id)]

for var in ['value_acc_x','value_acc_y','value_acc_z']:
    plt.figure(figsize=(15, 10))
    plot = sns.lineplot('time_ref', var, data = data, hue = 'id_', legend=False)
    plt.title(f'{var} over time')
    plot.set(xlabel = None, xticklabels = [])
    plot.tick_params(bottom = False)
    plt.show()





##### Data distribution of acceleration module
# There are extreme values but I'm not deleting it for now due to we can't conclude with this information that 
# there are errors in measurement (it might be related with the nature of such data).
# I must be careful with this because they can affect both the scoring and the model itself.

plt.figure(figsize=(15, 10))
sns.kdeplot(df_acc.mod_acc)
plt.xlabel("mod_acc")
plt.ylabel("density")
plt.title('Data distribution of acceleration module')
plt.show()





##### Relationship between acceleration module and jerk module

sns.jointplot(x = 'mod_acc', y = 'mod_jerk', data = df_acc, kind = 'reg', marginal_ticks = False, height = 15)
plt.show()





##### Data distribution of value_Hr and value_energy

for var in ['value_Hr', 'value_energy']:
    plt.figure(figsize=(15, 10))
    sns.kdeplot(df_energy['value_Hr'])
    plt.title(f'Data distribution of {var}')
    plt.xlabel(f'{var}')
    plt.ylabel('density')
    plt.show()





##### Relationship between value_Hr and value_energy

sns.jointplot(x = 'value_Hr', y = 'value_energy', data = df_energy, kind = 'reg', marginal_ticks = False, height = 15)
plt.show()


# # Data Preparation




##### Data Preparation
# This is the key part of the exercise. By reason of only having information of the activity and heart rate just in time intervals of certain 
# individuals, the pulses are a unique value. Then, the better we collect all the information of thephysical activity performed,
# more information the model will have to achieve a better accuracy.
# A difference is made between two sets of variables:
#    1st: the objective is to collect the maximum possible information on the distribution of acceleration and
#    jerk per individual (mean, median, std, etc.)
#    2nd: the objective is the system of frequencies and moments thanks to the transformation.  It's understood that by the nature of the problem 
#    this transformation can help to extract more information from the physical activity.
#    We leave only one observation per id in order to train the model with the energy data.

df_acc_agg = df_acc.groupby("id_").agg({'mod_acc': [np.sum, percentile(25), percentile(75), np.mean, np.median],
                                        'mod_jerk':[np.sum, percentile(25), percentile(75), np.mean, np.median ]}).reset_index()
df_acc_agg.columns = ['_'.join(col).strip() for col in df_acc_agg.columns.values]   
df_acc_agg.rename(columns={"id__":"id_"},inplace=True)
df_acc_agg





##### Merge of both datasets and save for modelling notebook

df_acc_final = pd.merge(df_acc_agg, df_energy, how="inner", on="id_")
print('Saving dataset...')
df_acc_final.to_csv('df_final.csv', index = False)
print('Done!')

