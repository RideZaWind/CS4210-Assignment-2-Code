#-------------------------------------------------------------------------
# AUTHOR: Adrian Alcoreza
# FILENAME: decision_tree_2.py
# SPECIFICATION: description of the program TODO: fill out
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append (row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    
    # Creating a dictionary that will map the original categorical feature values into numbers.
    transformed_values = {"Young": 1, "Presbyopic": 2, "Prepresbyopic": 3, 
                          "Myope": 1, "Hypermetrope": 2,
                          "Yes": 1, "No": 2,
                          "Normal": 1, "Reduced": 2}

    # Getting each except the last column of the db and replacing their values with numerical ones.
    X = [row[:-1] for row in dbTraining]
    X = [[transformed_values[old_value] for old_value in row] for row in X]

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    
    # Getting the last column of the db and replacing its values with numerical ones.
    Y = [row[-1] for row in dbTraining]
    Y = [1 if value == "Yes" else 2 for value in Y]
    
    accuracies = []

    #Loop your training and test tasks 10 times here
    for i in range (10):

        #Fitting the decision tree to the data setting max_depth=5
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
        clf = clf.fit(X, Y)

        #Read the test data and add this data to dbTest
        dbTest = []
        with open("contact_lens_test.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append (row)
            
        successes = 0

        for data in dbTest:
            #Transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            data = [transformed_values[old_value] for old_value in data]
            class_predicted = clf.predict([data[:4]])[0]

            #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            successes += 1 if class_predicted == data[4] else 0
            
        accuracies.append(successes / len(dbTest))

    #Find the average of this model during the 10 runs (training and test set)
    average_accuracy = sum(accuracies) / len(accuracies)

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f"Final accuracy when training on {ds}: {average_accuracy}")
