# Total Caloric Expenditure Prediction Model of Wearable Devices

#### Case study rules
The degree of implementation of the solution, the precision of the results obtained, the simplicity and explanatory nature of the solution found and its adjustment to the problem presented will be assessed.

It has been estimated that the time needed to solve this task is about 3 hours, but it is up to the candidate to dedicate more time if he or she considers it appropriate.

#### Case study statement
It's a wearable wrist device containing an accelerometer and a heart rate sensor. __The goal is to develop a system that will analyze the activity performed by the user based on the information from the sensors. For this purpose, the energy expenditure and activity level of the person should be predicted from the accelerometry and pulsation data recorded by the wearable device.__

A person's energy expenditure depends on their activity, metabolism and lifestyle, among others. For example, nowadays many smartwaches are able to approximate the user's energy expenditure based on their activity. Taking this into account, predict the energy expenditure of a person from their accelerometry and heart rate data. The files accs_x.csv, accs_y.csv and accs_z.csv contain the accelerations for each of the axes. The energy.csv file contains the energy data associated with the accelerometry and heart rate by time interval. 

The metric with which the prediction error will be measured is the MAPE (Median Absolute Percentage Error), expressed by the following formula:
<p align="center">
  <img src="https://github.com/sergiomlop/Total-Caloric-Expenditure-Prediction-Model-of-Wearable-Devices/blob/main/data/MAPE%20formula.png">
</p>

where T is the number of samples, 𝑦̂𝑡 is the actual value and 𝑦𝑡 is the prediction.  

In addition to the energy expenditure, for each time interval, the level of intensity of the activity carried out must be calculated. The classification of the intensity level is based on the metabolic equivalents or METS (kcal/kg*h) of the activity being a light activity < 3 METS, moderate 3 - 6 METS and intense > 6 METS. To estimate it, consider a person of 75 kg.
The wearable device has a battery and transmits the information to a Gateway in the user's home and this Gateway to a data storage and processing server. The system or algorithm developed must take into account the battery consumption of the wearable to reduce the expenditure to the minimum.

## Data

Datasets used for the problem and a pdf with a brief explanation of the variables.

	accs_x.csv
	accs_y.csv
	accs_z.csv
	energy.csv
	df_final.csv
	explicacion_variables.pdf

## Jupyter Notebooks

Python code in Jupyter Notebook format `.ipynb`

	1-Prediction-Model-Wearable-Devices-Exploratory-Data-Analysis.ipynb
	2-Prediction-Model-Wearable-Devices-Modelling.ipynb

## Python

Python code in Python Script format `.py`

	1-Prediction-Model-Wearable-Devices-Exploratory-Data-Analysis.py
	2-Prediction-Model-Wearable-Devices-Modelling.py

## Author
Sergio Mañanas López ([GitHub](https://github.com/sergiomlop) | [LinkedIn](https://www.linkedin.com/in/sergiomananaslopez/))  
