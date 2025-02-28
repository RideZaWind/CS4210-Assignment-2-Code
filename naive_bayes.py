#-------------------------------------------------------------------------
# AUTHOR: Adrian Alcoreza
# FILENAME: naive_bayes.py
# SPECIFICATION: Reads the file weather_training.csv and output the classification of each of the 10 instances from the file weather_test if the classification confidence is >= 0.75.
# FOR: CS 4210- Assignment #2
# TIME SPENT: Roughly 40 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
X = []
Y = []
dbTraining = []
dbTesting = []

with open("weather_training.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTraining.append (row)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]

# Creating a dictionary to translate the categorical values into numerical values.
transformed_values = {"Sunny": 1, "Overcast": 2, "Rain": 3,     # Outlook
                      "Cool": 1, "Mild": 2, "Hot": 3,           # Temperature
                      "Normal": 1, "High": 2,                   # Humidity
                      "Weak": 1, "Strong": 2                    # Wind
                      }
# Reading in columns into X, ignoring the indexing and class columns.
X = [row[1:-1] for row in dbTraining]
# Transforming the values in X into numerical values.
X = [[transformed_values[old_value] for old_value in row] for row in X]

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
# Reading the last column into Y and transforming its values.
Y = [row[-1] for row in dbTraining]
Y = [1 if value == "Yes" else 2 for value in Y]

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
with open("weather_test.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTesting.append(row)
            
# Removing the empty class column.
dbTesting = [row[:-1] for row in dbTesting]

#Printing the header os the solution
print(f"{"Day" : <10} {"Outlook" : <10} {"Temp." : <10}{"Humidity" : <10}{"Wind" : <10}{"PlayTennis" : <10} Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for row in dbTesting:
    prob_score_yes, prob_score_no = clf.predict_proba([[transformed_values[old_value] for old_value in row[1:]]])[0]
    
    if prob_score_yes > 0.75:
        print(f"{row[0] : <10} {row[1] : <10} {row[2] : <10}{row[3] : <10}{row[4] : <10}{"Yes" : <10}{prob_score_yes : .4f}")
    elif prob_score_no > 0.75:
        print(f"{row[0] : <10} {row[1] : <10} {row[2] : <10}{row[3] : <10}{row[4] : <10}{"No" : <10}{prob_score_no : .4f}")