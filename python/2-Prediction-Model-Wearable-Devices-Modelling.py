##### Import packages
# Basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modelling packages
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# To avoid warnings
import warnings
warnings.filterwarnings("ignore")





##### Import data
# Check the csv's path before running it

df_acc_final = pd.read_csv('df_final.csv')
df_acc_final





##### Creating Mean Absolute Percentage Error

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100





##### Format change to datetime on some energy columns

for col in ['date_Hr', 'startDate_energy', 'endDate_energy']:
    df_acc_final[col] = pd.to_datetime(df_acc_final[col])





##### Creating new variables based on energy data

df_acc_final["time_elapsed"] = (df_acc_final["startDate_energy"] - df_acc_final["date_Hr"]).astype('timedelta64[s]')
df_acc_final["day"] = df_acc_final.date_Hr.apply(lambda x: x.day)
df_acc_final["month"] = df_acc_final.date_Hr.apply(lambda x: x.month)
df_acc_final["hour"] = df_acc_final.date_Hr.apply(lambda x: x.hour)

df_acc_final.drop(['date_Hr', 'startDate_energy', 'endDate_energy','totalTime_energy'], axis=1, inplace=True)
df_acc_final.head()





##### To avoid problems while using MAPE, I multiply whole target x 10

df_acc_final.value_energy = df_acc_final.value_energy.apply(lambda x: x*10)


# # Modelling




##### Selecting all the columns to use to modelling (also the target)
# Before trying different models, it's important to keep in mind that the problem ask us for a model with not high computational 
# costs and that does not occupy much in the memory. In addition, it's valued the simplicity, clarity and explicitness.

features = list(df_acc_final)
for col in ['id_', 'value_energy']:
    features.remove(col)

print('Columns used on X:', features)





##### Creation of X and y

X = df_acc_final[features].values.astype('int')
y = df_acc_final['value_energy'].values.astype('int')





##### Creation of X and y split -- train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ## Decision Tree Regressor




##### Decision Tree Regressor
# This is a lightweight model related with memory usage and computationally

model = DecisionTreeRegressor()

params = {'criterion':['mae'],
        'max_depth': [4,5,6,7],
        'max_features': [7,8,9,10], 
        'max_leaf_nodes': [30,40,50], 
        'min_impurity_decrease' : [0.0005,0.001,0.005], 
        'min_samples_split': [2,4]}

# GridSearch
grid_solver = GridSearchCV(estimator = model, 
                   param_grid = params,
                   scoring = 'neg_median_absolute_error',
                   cv = 10,
                   refit = 'neg_median_absolute_error',
                   verbose = 0)

model_result = grid_solver.fit(X_train,y_train)

reg = model_result.best_estimator_
reg.fit(X,y)





##### Mean Absolute Percentage Error

yhat = reg.predict(X_test)
print("Mean Absolute Percentage Error = %.2f" %mean_absolute_percentage_error(yhat,y_test),'%')





##### Feature Importance

features_importance = reg.feature_importances_
features_array = np.array(features)
features_array_ordered = features_array[(features_importance).argsort()[::-1]]
features_array_ordered

plt.figure(figsize=(16,10))
sns.barplot(y = features_array, x = features_importance, orient='h', order=features_array_ordered[:50])

plt.show()


# ## Random Forest Regressor




##### Random Forest Regressor
# Random Forest model should lower the metric further because it maintains the bias and reduces the variance by making 
# combinations of models with low bias and high correlations but different from one value.
# The tree has a low bias but a high variance then I will try to combine models with low bias and that aren't completely correlated
# in order to to reduce the variance to its minimum value.

model = RandomForestRegressor()

params = {'bootstrap': [True],
        'criterion':['mae'],
        'max_depth': [8,10],
        'max_features': [10,12],
        'max_leaf_nodes': [10,20,30],
        'min_impurity_decrease' : [0.001,0.01],
        'min_samples_split': [2,4],
        'n_estimators': [10,15]}

# GridSearch
grid_solver = GridSearchCV(estimator = model, 
                   param_grid = params,
                   scoring = 'neg_median_absolute_error',
                   cv = 7,
                   refit = 'neg_median_absolute_error',
                   verbose = 0)

model_result = grid_solver.fit(X_train,y_train)

reg = model_result.best_estimator_
reg.fit(X,y)





##### Mean Absolute Percentage Error

yhat = reg.predict(X_test)
print("Mean Absolute Percentage Error = %.2f" %mean_absolute_percentage_error(yhat,y_test),'%')





##### Feature Importance

features_importance = reg.feature_importances_
features_array = np.array(features)
features_array_ordered = features_array[(features_importance).argsort()[::-1]]
features_array_ordered

plt.figure(figsize=(16,10))
sns.barplot(y = features_array, x = features_importance, orient='h', order=features_array_ordered[:50])

plt.show()


# ## SVM




##### SVM linear
# Although computationally it requires more effort, once the model is trained it takes up less memory space and it is very intuitive.
# After seeing graphs on EDA, it doesn't seem that the relations are linear but while trees have much flexibility, that algorithm is based on
# cuts by hyperplanes. I'll train different kernels for SVM to see if it fits better to the problem.

# Lineal Tuning

lineal_tuning = dict()
for c in [0.001,0.01, 1]:
    svr = SVR(kernel = 'linear', C = c)
    scores = cross_val_score(svr, X, y, cv = 5, scoring = 'neg_median_absolute_error')
    lineal_tuning[c] = scores.mean()

best_score = min(lineal_tuning, key = lineal_tuning.get)
print(f'Best score = {lineal_tuning[best_score]} is achieved with c = {best_score}')

reg = SVR(kernel = 'linear', C = best_score)
reg.fit(X_train, y_train)





##### Mean Absolute Percentage Error

yhat = reg.predict(X_test)
print("Mean Absolute Percentage Error = %.2f" %mean_absolute_percentage_error(yhat,y_test),'%')





##### SVM poly

reg = SVR(kernel = 'linear', C = 0.01)
reg.fit(X_train, y_train)





##### Mean Absolute Percentage Error

yhat = reg.predict(X_test)
print("Mean Absolute Percentage Error = %.2f" %mean_absolute_percentage_error(yhat,y_test),'%')





##### SVM radial

reg = SVR(kernel = 'rbf', C = 0.01, gamma = 0.1) 
reg.fit(X_train, y_train)





##### Mean Absolute Percentage Error

yhat = reg.predict(X_test)
print("Mean Absolute Percentage Error = %.2f" %mean_absolute_percentage_error(yhat,y_test),'%')


# # Activity Intensity




##### Activity Intensity
# In addition to calculate the energy expenditure, for each time interval, the level of intensity of the activity carried out must be calculated. 
# The classification of the intensity level is based on the metabolic equivalents or METS (kcal/kg*h) of the activity being:
# light activity < 3 METS, moderate 3 - 6 METS and intense > 6 METS. 
# To estimate it, I consider a person of 75 kg. The model chosen is the Random Forest Regressor which has the lowest MAPE.

reg = RandomForestRegressor(criterion='mae', max_depth=8, max_features=12,
                      max_leaf_nodes=30, min_impurity_decrease=0.001,
                      n_estimators=15)
reg.fit(X,y)

yhat = reg.predict(X)

ids = df_acc_final['id_'].to_frame()
ids['yhat'] = yhat
ids['METs'] = ids["yhat"] / (75 * 62 / 3600)

conditions = [(ids["METs"] < 3 ),((3 < ids["METs"]) & (ids["METs"] < 6)),(ids["METs"] > 6)]
names = ['ligera', 'moderada', 'intensa']
ids['intensidad'] = np.select(conditions, names)

ids





##### Conclusions and Future Work
# The substantial improvement that can be seen when we introduce the non-linearity of the model is relevant to deduce that
# the relationships between the variables and the target are not linear.
# The dataset doesn't have full potential to establish a clear model then more efforts should be made to collect all the information on physical
# activity, I suggest signal treatment variables such as Zero Crossing Rate, Spectral Centroid, Spectral Rolloff and MFCC - Mel-Frequency Cepstral Coefficients.
# Additional information about individuals such as age, sex and weight would help to improve the MAPE of final model.

# Time was decisive on this project (3-4h only) so some workstreams couldn't be done and would be important to have a look on.
# Extra efforts should be made in the selection of predictive variables to analyze the L1 and L2 error, otherwise we would be 
# losing explicitness, memory and battery. 

