from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

def BuildNewStructureData(array_vectors,list_dataset):
    Structure_data = list()#list of items

    for x in range(len(list_dataset[1])):

        new_item=list()#each items is composed as
        new_item.append(list_dataset[0][x])#id
        new_item.append(array_vectors[x])#vector

        new_item.append((list_dataset[5][x]))#relevant- not relevant
        if(new_item[2]=="not relevant"):
            new_item[2]=1
        else:
            new_item[2]=0
        Structure_data.append(new_item)
    print(Structure_data)
    return Structure_data
def SVMalgorithm(lists):
    vectors = list()
    target = list()
    for x in range(len(lists)):

       vectors.append(lists[x][1])
       target.append(lists[x][2])


    X_train, X_test, y_train, y_test = train_test_split(vectors,target, test_size=0.3,
                                                        random_state=109)  # 70% training and 30% test
    clf = svm.SVC(kernel='linear')
    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print("Metrics\n")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(y_test, y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test, y_pred))
    return
