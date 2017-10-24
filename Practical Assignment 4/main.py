from scipy.io import loadmat
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np


def show_image(X, i):
	plt.imshow(np.reshape(X[i], newshape=[28, 28]).transpose())
	plt.show()


def split_data(x, y, test_size=0.2, shuffle=True):
	x_6 = x[0:50]
	x_9 = x[50:100]
	y_6 = y[0:50]
	y_9 = y[50:100]

	x_6_train, x_6_test, y_6_train, y_6_test = train_test_split(x_6, y_6, test_size=test_size, shuffle=shuffle)
	x_9_train, x_9_test, y_9_train, y_9_test = train_test_split(x_9, y_9, test_size=test_size, shuffle=shuffle)

	x_train = np.vstack((x_6_train, x_9_train))
	x_test = np.vstack((x_6_test, x_9_test))
	y_test = np.vstack((y_6_test, y_9_test))
	y_train = np.vstack((y_6_train, y_9_train))

	return x_train, x_test, y_train, y_test


def exercise_1(plot=False):
	raw = loadmat("69dataset.mat")

	Y = raw['Y']
	X = raw['X']

	X = normalize(X)
	Y = normalize(Y)

	X_train, X_test, Y_train, Y_test = split_data(X, Y, shuffle=False)

	reg = Ridge(alpha=10**-6)
	reg.fit(Y_train, X_train)

	X_pred = reg.predict(Y_test)

	if plot:
		for i in range(20):
			show_image(X_pred, i)

	return X_test, X_pred


def exercise_2(plot=False):
	raw = loadmat("69dataset.mat")

	Y = raw['Y']
	X = raw['X']
	X_prior = raw['prior']

	X = normalize(X)
	Y = normalize(Y)
	X_prior = normalize(X_prior)

	X_train, X_test, Y_train, Y_test = split_data(X, Y, shuffle=False)

	reg = Ridge(alpha=10 ** -6)
	reg.fit(X_train, Y_train)

	sigma_prior = np.dot(X_prior.T, X_prior) / (len(X_prior) - 1)
	sigma_prior = sigma_prior + np.multiply(np.identity(len(sigma_prior)), 10**-6)  # regularization
	# sigma_prior = np.cov(X_prior.T)
	sigma = np.multiply(np.identity(reg.coef_.shape[0]), 10**-3)

	if plot:
		plt.imshow(sigma_prior.transpose())
		plt.show()

	sigma_inv = np.linalg.inv(sigma)

	print("B: {} \n sigma_prior_inv: {} \n sigma_inv: {} \n B.T: {} \n".format(reg.coef_.shape, np.linalg.inv(sigma_prior).shape, sigma_inv.shape, reg.coef_.T.shape))

	u_post = np.linalg.inv(sigma_prior)
	u_post = u_post + np.dot(np.dot(reg.coef_.T, sigma_inv), reg.coef_)
	u_post = np.linalg.inv(u_post)
	u_post = np.dot(u_post, reg.coef_.T)
	u_post = np.dot(u_post, sigma_inv)
	res = [np.dot(u_post, y) for y in Y_test]

	if plot:
		for i in range(len(res)):
			show_image(res, i)

	return res


def compare(X, X1, X2):
	for i in range(X1.shape[0]):
		plt.subplot(131).set_title("Real Image")
		plt.imshow(np.reshape(X[i], newshape=[28, 28]).transpose())
		plt.subplot(132).set_title("Ridge Regression")
		plt.imshow(np.reshape(X1[i], newshape=[28, 28]).transpose())
		plt.subplot(133).set_title("Bayes")
		plt.imshow(np.reshape(X2[i], newshape=[28, 28]).transpose())
		plt.show()


x, x1 = exercise_1()
x2 = exercise_2()
compare(x, x1, x2)
