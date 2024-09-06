import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################


def my_fit(X_train, y_train):
    ################################
    #  Non Editable Region Ending  #
    ################################

    # Use this method to train your model using training CRPs
    # X_train has 32 columns containing the challeenge bits
    # y_train contains the responses

    # THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
    # If you do not wish to use a bias term, set it to 0
    classifier = LogisticRegression(penalty='l2', C=16.0, tol=0.0001)
    X_train = my_map(X_train)
    classifier.fit(X_train, y_train)
    w = classifier.coef_[0]
    b = classifier.intercept_[0]
    return w.T, b


################################
# Non Editable Region Starting #
################################
def my_map(X):
    ################################
    #  Non Editable Region Ending  #
    ################################

    # Use this method to create features.
    # It is likely that my_fit will internally call my_map to create features for train points

    X_train = np.zeros((X.shape[0], X.shape[1]+1))
    X_train[:, X.shape[1]] = 1
    for j in range(X.shape[1]-1, -1, -1):
        X_train[:, j] = X_train[:, j+1]*(1-2*X[:, j])

    X_train_T = np.transpose(X_train)  # transpose of X_train
    X_model_temp = khatri_rao(X_train_T, X_train_T)

    rows_to_delete = []
    for i in range(33):
        rows_to_delete.append(33*i+i)
        for j in range(i+1, 33):
            rows_to_delete.append(33*i+j)

    X_model = np.delete(X_model_temp, rows_to_delete, axis=0)
    return np.transpose(X_model)
