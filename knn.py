#-------------------------------------------------------------------------
# AUTHOR: Adrian Alcoreza
# FILENAME: knn.py
# SPECIFICATION: Reads the file email_classification.csv and compute the LOO-CV error rate for a 1NN classifier on the spam/ham classification task.
# FOR: CS 4210- Assignment #2
# TIME SPENT: Roughly 20 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []
successful_classifications = 0

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#Loop your data to allow each instance to be your test set
for i in db:

    #Add the training features to the 2D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 3], [2, 1,], ...]].
    #Convert each feature value to float to avoid warning messages
    X = [row[:-1] for row in db if row != i]
    X = [[float(entry) for entry in row] for row in X]

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    Y = [row[-1:][0] for row in db if row != i]
    # With Spam = 1 and Ham = 2
    Y = [1. if classification == "spam" else 2. for classification in Y]

    #Store the test sample of this iteration in the vector testSample
    testSample = i
    testSample[-1] = 1. if testSample[-1] == "spam" else 2.
    testSample = [float(entry) for entry in testSample]

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    class_predicted = clf.predict([testSample[:-1]])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    successful_classifications += 1 if class_predicted == testSample[-1] else 0

#Print the error rate
print(f"Error Rate: {100 * (1 - (successful_classifications / len(db))) : .2f}%")
