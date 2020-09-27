import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statistics import mean
import math

# Hand-Created Linear Regression Class

class Linear_Regression_Model:
	def __init__(self, intercept = 0, slope = 0):
		self.intercept = intercept
		self.slope = slope


	def fit(self, x_array, y_array):
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


	def predict(self, x, int_adjust = 0, slp_adjust = 0):
		y = self.intercept + int_adjust + ((self.slope + slp_adjust) * x)
		return y


	def Standard_Error(self, x_array, y_array, int_adjust = 0, slp_adjust = 0):
		total_error = 0
		for i in range(0, len(x_array)):
			total_error += ((y_array[i] - self.predict(x_array[i], int_adjust, slp_adjust))) ** 2
		error = math.sqrt(total_error / len(x_array))
		return error


