# Load Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap

from pandas.plotting import scatter_matrix

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
data = load_digits()
X = data.data  # extracting the flattened version of the digits image
y = data.target  # extracting the actual target value for that digit image

# Explore dataset
print(data.images.shape)  # determine shape of the images
print(dir(data), '\n')  # descriptions
print(X.shape)  # shape of the flattened images
print(y.shape)  # shape of the target digits

# Visualize dataset
sns.set()
fig, axes = plt.subplots(10, 10, figsize=(8, 8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(data.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(data.target[i]))
plt.show()

# Dataset dimensionality
isomapModel = Isomap(n_components=2)  # initialize Isomap model
isomapModel.fit(data.data)  # fit model to the higher dimensioned data
data_transformed = isomapModel.transform(data.data)  # transform from 2 dimensions

print(data_transformed.shape)  # show the shape of the transformed data
plt.scatter(data_transformed[:, 0], data_transformed[:, 1], c=data.target, edgecolors='none', alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()

# Model dataset
validation_size = 0.2
seed = 7
X_training_set, X_validation_set, Y_training_set, Y_validation_set = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

# Test Harness factors
seed = 7
scoring = 'accuracy'  # criteria used to validate the algorithm to be spot-checked

# Test Harness Setup
# to spot the following classification problem algorithms with default settings
# -   Logistic Regression [LR]
# -   Linear Discriminant Analysis [LDA]
# -   K-Nearest Neighbors (KNN)
# -   Classification and Regression Trees [CART]
# -   Gaussian Naive Bayes [NB]
# -   Support Vector Machines [SVM]

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))  # append a tuple of the algorithm and algorithm description
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NG', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
results = []
names = []
for name, model in models:  # iterate through the models so as to run cross-validation for each model i.e. each algorithm
    kfold = model_selection.KFold(n_splits=10, random_state=seed)  # initialize 10-fold cross-validation data splits
    cross_validation_results = model_selection.cross_val_score(model, X_training_set, Y_training_set, cv=kfold, scoring=scoring)
    results.append(cross_validation_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cross_validation_results.mean(), cross_validation_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Make Predictions based on selecting KNN as the algorithm
knn = KNeighborsClassifier()
knn.fit(X_training_set, Y_training_set)  # to train
classification_predictions = knn.predict(X_validation_set)  # to test/validate
print(accuracy_score(Y_validation_set, classification_predictions))
print(classification_report(Y_validation_set, classification_predictions))

# Compare Prediction accuracy
cmValues = confusion_matrix(Y_validation_set, classification_predictions)
print(cmValues)  # show where predictions do not match up to the target value
sns.heatmap(cmValues, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('target value')
plt.show()

fig, axes = plt.subplots(10, 10, figsize=(8, 8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
test_images = X_validation_set.reshape(-1, 8, 8)
for i, ax in enumerate(axes.flat):
    ax.imshow(test_images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(classification_predictions[i]), transform=ax.transAxes, color='green' if (Y_validation_set[i] == classification_predictions[i]) else 'red')
plt.show()
