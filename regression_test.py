# For this project, we're also going to create a dataset from scratch. 
# It will have two variables, labled indepenent variable, and dependent variable. with 1,000,000 datapoints
# The Independent Variable will be a set of random generations within a normal distribution centered at 100 with dtandard deviation 10
# The Dependant Variable will be randomly generated with mean 100 minus 0.5x


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
for i in range(0, 10000):
	x = np.random.normal(100, 10)
	ind.append(x)
	y = 0.5*(x-100) + np.random.normal(100, .01)
	dep.append(y)




##### TEST of hand-created linear regression model ####################
test = Linear_Regression_Model()
test.fit(ind, dep)


print("Hand Created Slope and Intercept")
print("Intercept: " + str(round(test.intercept, 5)))
print("Slope " + str(round(test.slope, 5)))


plt.plot(ind,dep,"o")
#plt.show()
#######################################################################
print(test.predict(125))
print(test.Standard_Error(ind, dep))

###### Putting data into a Pandas dataframe for use with SKLearn's model 
#df = pd.DataFrame(columns=['ind','dep'])
df = pd.DataFrame()
df['ind'] = pd.Series(ind)
df['dep'] = pd.Series(dep)
#print(df.head())

