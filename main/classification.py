#from pandas.tests.extension.numpy_.test_numpy_nested import np
from mlxtend.data import iris
from mlxtend.preprocessing import shuffle_arrays_unison
from pandas.tests.extension.numpy_.test_numpy_nested import np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from matplotlib.colors import *
import pandas as pd


def BuildNewStructureData(array_vectors, list_dataset):
    Structure_data = list()#list of items


    for x in range(len(list_dataset[1])):

        new_item=list()#each items is composed as
        new_item.append(list_dataset[0][x])#id
        new_item.append(array_vectors[x])#vector

        new_item.append((list_dataset[5][x]))#relevant- not relevant
        new_item.append((list_dataset[4][x]))#-3 Flood 1-2 Earthquake

        if(new_item[2]=="not relevant"):
            new_item[2]=1
        else:
            new_item[2]=0
        if(new_item[3]==0 | new_item[3]==3):
            new_item[3]=0#0=Flood
        else:
            new_item[3]=1 #1"Eartquake"
        Structure_data.append(new_item)

    return Structure_data

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def SVMalgorithmKind(lists, gamma=1):
    vectors = list()
    target = list()
    for x in range(len(lists)):

       vectors.append(lists[x][1])
       target.append(lists[x][3])


    X_train, X_test, y_train, y_test = train_test_split(vectors,target, test_size=0.3,
                                                        random_state=50)  # 70% training and 30% test


    clf = svm.SVC(kernel='linear', C=gamma)
    # Train the model using the training sets
    clf.fit(X_train, y_train)
    #plot_decision_function(X_train, y_train, X_test, y_test, clf)
    # Predict the response for test dataset
    y_pred = clf.predict(X_test)


    print("Metrics\n")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


    #plt.scatter(X_train, y_train, label="stars", color="green",
     #           marker="*", s=10)
   # plt.scatter(X_test, y_test, label="croci", color="red",
      #          marker="X", s=10)
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision element y:", metrics.precision_score(y_test, y_pred))


    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test, y_pred))
    probs = y_pred
    auc = roc_auc_score(y_test, probs)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    plot_roc_curve(fpr, tpr)
    print('AUC: %.2f' % auc)

    #plt.show()
    return



def SVMalgorithmRelevantNotRelevant(lists):
    vectors = list()
    target = list()
    for x in range(len(lists)):

       vectors.append(lists[x][1])
       target.append(lists[x][2])

#Split the data into train and test sub-datasets.
    X_train, X_test, y_train, y_test = train_test_split(vectors,target, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test
    clf = svm.SVC(kernel='linear')
    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test) #Predict probabilities for the test data

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    (tn, fp, fn, tp) = confusion_matrix.ravel()
    accuracy = (tn + tp) / (tp + tn + fp + fn)
    print("CLASSIFICATION REPORT ====>")
    print("Accuracy: %0.2f" % (accuracy))
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix)


    print("Metrics\n")
   # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision element y:", metrics.precision_score(y_test, y_pred))

    '''In the event where both the class distribution simply mimic each other, AUC is 0.5. In other words, our model is 50% accurate for instances and their classification. The model has no discrimination capabilities at all in this case'''
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test, y_pred))
    probs = y_pred[:]
    y_test=y_test[:]
    auc = roc_auc_score(y_test, probs)
    print('AUC: %.2f' % auc)
    fpr, tpr, thresholds = roc_curve(y_test, probs) #Get the ROC Curve.
    plot_roc_curve(fpr, tpr) # Plot ROC Curve using our defined function

    return


def BayesalgorithmRelevantNotRelevant(lists):
    vectors = list()
    target = list()
    for x in range(len(lists)):

       vectors.append(lists[x][1])
       target.append(lists[x][2])


    X_train, X_test, y_train, y_test = train_test_split(vectors,target, test_size=0.3,
                                                        random_state=109)  # 70% training and 30% test
    model = MultinomialNB().fit(X_train, y_train)

    predicted = model.predict(X_test)

    print(np.mean(predicted == y_test))
    print(confusion_matrix(y_test, predicted))

    # Predict the response for test dataset

    print("Metrics\n")
    print(confusion_matrix(y_test, predicted))
    print(classification_report(y_test, predicted))
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, predicted))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision element y:", metrics.precision_score(y_test,  predicted))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test,  predicted))
    probs = predicted[:]
    y_test = y_test[:]
    auc = roc_auc_score(y_test, probs)
    print('AUC: %.2f' % auc)
    fpr, tpr, thresholds = roc_curve(y_test, probs)  # Get the ROC Curve.
    plot_roc_curve(fpr, tpr)  # Plot ROC Curve using our defined function


    return

def BayesalgorithmKind(lists):
    vectors = list()
    target = list()
    for x in range(len(lists)):

       vectors.append(lists[x][1])
       target.append(lists[x][3])


    X_train, X_test, y_train, y_test = train_test_split(vectors,target, test_size=0.3,
                                                        random_state=109)  # 70% training and 30% test
    model = MultinomialNB().fit(X_train, y_train)

    predicted = model.predict(X_test)

    print(np.mean(predicted == y_test))
    print(confusion_matrix(y_test, predicted))

    # Predict the response for test dataset

    print("Metrics\n")
    print(confusion_matrix(y_test, predicted))
    print(classification_report(y_test, predicted))
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, predicted))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision element y:", metrics.precision_score(y_test,  predicted))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test,  predicted))
#https://medium.com/data-py-blog/kernel-svm-in-python-a8fae37908b9
    probs = predicted[:]
    y_test = y_test[:]
    auc = roc_auc_score(y_test, probs)
    print('AUC: %.2f' % auc)
    fpr, tpr, thresholds = roc_curve(y_test, probs)  # Get the ROC Curve.
    plot_roc_curve(fpr, tpr)  # Plot ROC Curve using our defined function


    return