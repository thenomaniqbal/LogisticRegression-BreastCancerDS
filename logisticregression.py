#------------------------------------------------------
# Name      : Gogul Ilango
# Purpose   : Implement Logistic Regression from scratch
#             using python and numpy
# Variants  : 1. LR without L2 regularization
#             2. LR with L2 regularization
# Libraries : numpy, scikit-learn
#------------------------------------------------------

# organize imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ignore all warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# load scikit-learn's breast cancer dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

print(data.keys())
print("No.of.data points (rows) : {}".format(len(data.data)))
print("No.of.features (columns) : {}".format(len(data.feature_names)))
print("No.of.classes            : {}".format(len(data.target_names)))
print("Class names              : {}".format(list(data.target_names)))

# view the datatype of each column
df = pd.DataFrame(data.data)
print(df.dtypes)

# split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.20, random_state=9)

print("X_train : " + str(X_train.shape))
print("y_train : " + str(y_train.shape))
print("X_test : " + str(X_test.shape))
print("y_test : " + str(y_test.shape))

#---------------------------------------------
# logistic regression without regularization
#---------------------------------------------
def sigmoid(score):
	return (1 / (1 + np.exp(-score)))

def predict_probability(features, weights):
	score = np.dot(features, weights)
	return sigmoid(score)

def feature_derivative(errors, feature):
	derivative = np.dot(np.transpose(errors), feature)
	return derivative

def l2_feature_derivative(errors, feature, weight, l2_penalty, feature_is_constant):
	derivative = np.dot(np.transpose(errors), feature)
	if not feature_is_constant:
		derivative -= 2 * l2_penalty * weight
	return derivative

def compute_log_likelihood(features, labels, weights):
	indicators = (labels==+1)
	scores     = np.dot(features, weights)
	ll         = np.sum((np.transpose(np.array([indicators]))-1)*scores - np.log(1. + np.exp(-scores)))
	return ll

def l2_compute_log_likelihood(features, labels, weights, l2_penalty):
	indicators = (labels==+1)
	scores     = np.dot(features, weights)
	ll         = np.sum((np.transpose(np.array([indicators]))-1)*scores - np.log(1. + np.exp(-scores))) - (l2_penalty * np.sum(weights[1:]**2))
	return ll

# logistic regression without L2 regularization
def logistic_regression(features, labels, lr, epochs):

	# add bias (intercept) with features matrix
	bias      = np.ones((features.shape[0], 1))
	features  = np.hstack((bias, features))

	# initialize the weight coefficients
	weights = np.zeros((features.shape[1], 1))

	logs = []

	# loop over epochs times
	for epoch in range(epochs):

		# predict probability for each row in the dataset
		predictions = predict_probability(features, weights)

		# calculate the indicator value
		indicators = (labels==+1)

		# calculate the errors
		errors = np.transpose(np.array([indicators])) - predictions

		# loop over each weight coefficient
		for j in range(len(weights)):

			# calculate the derivative of jth weight cofficient
			derivative = feature_derivative(errors, features[:,j])
			weights[j] += lr * derivative

		# compute the log-likelihood
		ll = compute_log_likelihood(features, labels, weights)
		logs.append(ll)

	import matplotlib.pyplot as plt
	x = np.linspace(0, len(logs), len(logs))
	fig = plt.figure()
	plt.plot(x, logs)
	fig.suptitle('Training the classifier (without L2)')
	plt.xlabel('Epoch')
	plt.ylabel('Log-likelihood')
	fig.savefig('train_without_l2.jpg')
	plt.show()

	return weights

# logistic regression with L2 regularization
def l2_logistic_regression(features, labels, lr, epochs, l2_penalty):

	# add bias (intercept) with features matrix
	bias      = np.ones((features.shape[0], 1))
	features  = np.hstack((bias, features))

	# initialize the weight coefficients
	weights = np.zeros((features.shape[1], 1))

	logs = []

	# loop over epochs times
	for epoch in range(epochs):

		# predict probability for each row in the dataset
		predictions = predict_probability(features, weights)

		# calculate the indicator value
		indicators = (labels==+1)

		# calculate the errors
		errors = np.transpose(np.array([indicators])) - predictions

		# loop over each weight coefficient
		for j in range(len(weights)):

			isIntercept = (j==0)

			# calculate the derivative of jth weight cofficient
			derivative = l2_feature_derivative(errors, features[:,j], weights[j], l2_penalty, isIntercept)
			weights[j] += lr * derivative

		# compute the log-likelihood
		ll = l2_compute_log_likelihood(features, labels, weights, l2_penalty)
		logs.append(ll)

	import matplotlib.pyplot as plt
	x = np.linspace(0, len(logs), len(logs))
	fig = plt.figure()
	plt.plot(x, logs)
	fig.suptitle('Training the classifier (with L2)')
	plt.xlabel('Epoch')
	plt.ylabel('Log-likelihood')
	fig.savefig('train_with_l2.jpg')
	plt.show()

	return weights

# logistic regression without regularization
def lr_without_regularization():
	# hyper-parameters
	learning_rate = 1e-7
	epochs        = 500

	# perform logistic regression and get the learned weights
	learned_weights = logistic_regression(X_train, y_train, learning_rate, epochs)

	# make predictions using learned weights on testing data
	bias_train     = np.ones((X_train.shape[0], 1))
	bias_test      = np.ones((X_test.shape[0], 1))
	features_train = np.hstack((bias_train, X_train))
	features_test  = np.hstack((bias_test, X_test))

	test_predictions  = (predict_probability(features_test, learned_weights).flatten()>0.5)
	train_predictions = (predict_probability(features_train, learned_weights).flatten()>0.5)
	print("Accuracy of our LR classifier on training data: {}".format(accuracy_score(np.expand_dims(y_train, axis=1), train_predictions)))
	print("Accuracy of our LR classifier on testing data: {}".format(accuracy_score(np.expand_dims(y_test, axis=1), test_predictions)))

	# using scikit-learn's logistic regression classifier
	model = LogisticRegression(random_state=9)
	model.fit(X_train, y_train)
	sk_test_predictions  = model.predict(X_test)
	sk_train_predictions = model.predict(X_train)
	print("Accuracy of scikit-learn's LR classifier on training data: {}".format(accuracy_score(y_train, sk_train_predictions)))
	print("Accuracy of scikit-learn's LR classifier on testing data: {}".format(accuracy_score(y_test, sk_test_predictions)))

	#visualize_weights(np.squeeze(learned_weights), 'weights_without_l2.jpg')

# logistic regression with regularization
def lr_with_regularization():
	# hyper-parameters
	learning_rate = 1e-7
	epochs        = 300000
	l2_penalty    = 0.001

	# perform logistic regression and get the learned weights
	learned_weights = l2_logistic_regression(X_train, y_train, learning_rate, epochs, l2_penalty)

	# make predictions using learned weights on testing data
	bias_train     = np.ones((X_train.shape[0], 1))
	bias_test      = np.ones((X_test.shape[0], 1))
	features_train = np.hstack((bias_train, X_train))
	features_test  = np.hstack((bias_test, X_test))

	test_predictions  = (predict_probability(features_test, learned_weights).flatten()>0.5)
	train_predictions = (predict_probability(features_train, learned_weights).flatten()>0.5)
	print("Accuracy of our LR classifier on training data: {}".format(accuracy_score(np.expand_dims(y_train, axis=1), train_predictions)))
	print("Accuracy of our LR classifier on testing data: {}".format(accuracy_score(np.expand_dims(y_test, axis=1), test_predictions)))

	# using scikit-learn's logistic regression classifier
	model = LogisticRegression(random_state=9)
	model.fit(X_train, y_train)
	sk_test_predictions  = model.predict(X_test)
	sk_train_predictions = model.predict(X_train)
	print("Accuracy of scikit-learn's LR classifier on training data: {}".format(accuracy_score(y_train, sk_train_predictions)))
	print("Accuracy of scikit-learn's LR classifier on testing data: {}".format(accuracy_score(y_test, sk_test_predictions)))

	visualize_weights(np.squeeze(learned_weights), 'weights_with_l2.jpg')

# visualize weight coefficients
def visualize_weights(weights, title):
	import matplotlib.pyplot as plt
	x = np.linspace(0, len(weights), len(weights))

	fig = plt.figure()
	plt.bar(x, weights, align='center', alpha=0.5)
	plt.xlabel("Weight Index (Feature Column Number)")
	plt.ylabel("Weight Coefficient")
	plt.title('Visualizing Weights')
	plt.tight_layout()
	fig.savefig(title)

	plt.show()

lr_with_regularization()