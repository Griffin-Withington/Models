import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statistics import mean

# Hand-Created Linear Regression Class

class Linear_Regression_Model:
	def __init__(self, intercept = None, slope = None):
		self.intercept = intercept
		self.slope = slope


	def fit(self, x_array, y_array):
		# Check if arrays are the same length
		x_mean = float(mean(x_array))
		y_mean = float(mean(y_array))

		if len(x_array) != len(y_array):
			print("Bad Arrays")

		else:
			alpha = 0
			beta = 0

			for i in range(0, len(x_array)):
				alpha += (x_array[i]-x_mean)*(y_array[i]-y_mean)
				beta += (x_array[i]-x_mean) ** 2

		slope = alpha/beta
		intercept = y_mean - (slope*x_mean)

		self.intercept = intercept
		self.slope = slope