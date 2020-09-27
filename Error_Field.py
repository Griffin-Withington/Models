# Generation of an Error Field


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statistics import mean

from Linear_Regression import Linear_Regression_Model

ind = []
dep = []
for i in range(0, 100):
	x = np.random.normal(100, 10)
	ind.append(x)
	y = 0.5*(x-100) + np.random.normal(100, 10)
	dep.append(y)




##### TEST of hand-created linear regression model ####################
test = Linear_Regression_Model()
test.fit(ind, dep)

##### Gonna Create a DataFrame with the rows being changes in intercept, collumns being changes in slope

error_field = pd.DataFrame()
rows = []
for i in range(-100, 101):
	rows.append(i/10.0)
	col = []
	slope_change = i/100.0
	for j in range(-100, 101):
		int_change = j/10.0
		error = test.Standard_Error(ind, dep, int_change, slope_change)
		col.append(error)
	col_name = str(slope_change)
	if slope_change > 0:
		col_name = '+' + col_name
	error_field[col_name] = pd.Series(col)






error_field.index = pd.Series(rows)


print(error_field.head(21))

plt.pcolor(error_field)
#plt.xticks(error_field.index)
#plt.yticks(error_field.columns)
plt.show()


error_field.to_csv(r'C:\Users\Griffin\Desktop\Q4 2020\Linear Regression\Hand Made\error_field.csv', index=True, header=True)
