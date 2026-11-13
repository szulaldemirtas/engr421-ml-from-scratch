import numpy as np
import pandas as pd

def safelog(x):
    return(np.log(x + 1e-300))



X_train = np.genfromtxt("pendigits_sta16_train.csv", delimiter = ",", dtype = int)
y_train = np.genfromtxt("pendigits_label_train.csv", delimiter = ",", dtype = int)
X_test = np.genfromtxt("pendigits_sta16_test.csv", delimiter = ",", dtype = int)
y_test = np.genfromtxt("pendigits_label_test.csv", delimiter = ",", dtype = int)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# STEP 3
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below

    K = 10
    class_priors =[np.mean(y == (c + 1)) for c in range(K)]
    
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 4
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D)
def estimate_success_probabilities(X, y):
    # your implementation starts below

    K = 10
    D = X.shape[1]

    P = np.zeros((K, D))

    for c in range(K):
        for d in range(D):
            P[c,d] = np.mean(X[(y == (c + 1), d)]) 

    # your implementation ends above
    return(P)

P = estimate_success_probabilities(X_train, y_train)
print(P)


# STEP 5
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, P, class_priors):
    # your implementation starts below
    K = 10
    D = X.shape[1]
    N = X.shape[0]
    epsilon = 1e-10
    
    score_values = np.zeros((N, K))
    
    for c in range(K):
        for n in range(N):
           score_values[n][c] = sum([X[n][d] * (safelog(P[c][d])) for d in range(D)]) + \
                                sum((1 - X[n][d]) * (safelog(1 - P[c][d])) for d in range(D)) + \
                                safelog(class_priors[c]) 

    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, P, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, P, class_priors)
print(scores_test)



# STEP 6
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    K = 10
    confusion_matrix = np.zeros((K, K))    
    y_pred = [np.argmax(score_list) + 1 for score_list in scores]

    for i in range(K):
        for j in range(K):
            confusion_matrix[i][j] = np.sum(np.logical_and(y_truth == (j + 1), np.array(y_pred) == (i + 1)))

    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)
print("Training accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_train)) / np.sum(confusion_train)))
confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
print("Test accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_test)) / np.sum(confusion_test)))
