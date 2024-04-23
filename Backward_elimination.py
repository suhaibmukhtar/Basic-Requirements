
# for regression
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Load the Boston housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
lr = LinearRegression()

# Initialize the SequentialFeatureSelector for backward elimination
sfs_backward = SFS(lr,
                   k_features='best',
                   forward=False, #for forward selection keep it true
                   floating=False,
                   scoring='neg_mean_squared_error', #r2, accuracy_score
                   cv=5)

# Fit the SequentialFeatureSelector to the training data
sfs_backward = sfs_backward.fit(X_train, y_train)

# Get the selected feature names and indices
selected_features = list(sfs_backward.k_feature_names_)
selected_feature_indices = list(sfs_backward.k_feature_idx_)

# Print the selected features
print("Selected features:", selected_features)
print("Selected feature indices:", selected_feature_indices)

# Transform the training and testing data to include only selected features
X_train_selected = X_train.iloc[:, selected_feature_indices]
X_test_selected = X_test.iloc[:, selected_feature_indices]

# Train a new model using only the selected features
lr_selected = LinearRegression()
lr_selected.fit(X_train_selected, y_train)

# Evaluate the model on the test set
test_score = lr_selected.score(X_test_selected, y_test)
print("R^2 score on test set with selected features:", test_score)

##for classification
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
logreg = LogisticRegression(solver='lbfgs', multi_class='auto')

# Initialize the SequentialFeatureSelector for backward elimination
sfs_backward = SFS(logreg,
                   k_features='best',
                   forward=False,
                   floating=False,
                   scoring='accuracy',
                   cv=5)

# Fit the SequentialFeatureSelector to the training data
sfs_backward = sfs_backward.fit(X_train, y_train)

# Get the selected feature names and indices
selected_features = list(sfs_backward.k_feature_names_)
selected_feature_indices = list(sfs_backward.k_feature_idx_)

# Print the selected features
print("Selected features:", selected_features)
print("Selected feature indices:", selected_feature_indices)

# Transform the training and testing data to include only selected features
X_train_selected = X_train.iloc[:, selected_feature_indices]
X_test_selected = X_test.iloc[:, selected_feature_indices]

# Train a new model using only the selected features
logreg_selected = LogisticRegression(solver='lbfgs', multi_class='auto')
logreg_selected.fit(X_train_selected, y_train)

# Predict on the test set using the model with selected features
y_pred = logreg_selected.predict(X_test_selected)

# Evaluate the model's accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set with selected features:", accuracy)
