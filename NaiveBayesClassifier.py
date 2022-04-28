# Peter Brede Naive Bayes Classifier

# Import Statements
import sys
import pandas as pd
import numpy as np
import math
from random import sample
import random
from itertools import combinations



# First Format data
features = ["location", "minTemp", "maxTemp",
    "Rainfall", "Evaporation", "Sunshine", "WindGustDir", 
    "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am", "WindSpeed3pm", 
    "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", 
    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RainToday", "RainTomorrow"]
traindata = np.array(pd.read_csv('training.txt', names=features).to_numpy(), ndmin=2)
testdata = np.array(pd.read_csv('private_testing_input.txt', names=features).to_numpy(), ndmin=2)

# Data Cleaning
locations = list(np.unique(traindata[:,0]))
directions = list(np.unique(traindata[:,6]))
for row in range(traindata.shape[0]):
    for col in range(traindata.shape[1]):
        ele = traindata[row][col]
        if (ele in locations):
            traindata[row][col] = locations.index(ele)
        if (ele in directions):
            traindata[row][col] = directions.index(ele)
        if (ele == " Yes"):
            traindata[row][col] = 1
        if (ele == " No"):
            traindata[row][col] = 0

for row in range(testdata.shape[0]):
    for col in range(testdata.shape[1]):
        ele = testdata[row][col]
        if (ele in locations):
            testdata[row][col] = locations.index(ele)
        if (ele in directions):
            testdata[row][col] = directions.index(ele)
        if (ele == " Yes"):
            testdata[row][col] = 1
        if (ele == " No"):
            testdata[row][col] = 0

        
# allRain is all rows with RainTomorrow = yes
count = 0
for i in range(traindata.shape[0]):
    if (traindata[i][21] == 1):
        count += 1

allRain = np.zeros((count,21))
notRain = np.zeros((traindata.shape[0] - count,21))

ind1 = 0
ind2 = 0
for i in range(traindata.shape[0]):
    if (traindata[i][21] == 1):
        allRain[ind1,:] = (traindata[i,:21])
        ind1 += 1
    else:
        notRain[ind2,:] = (traindata[i,:21])
        ind2 += 1

# Finding means and std devs of training columns
def mean(list_num):
    return sum(list_num)/float(len(list_num))

def std_dev(list_num):
    avg = mean(list_num)
    var = sum([(val - avg)**2 for val in list_num]) / float(len(list_num) - 1)
    return(math.sqrt(var))

imp_data_Rain = [[] for _ in range(21)]
imp_data_NotRain = [[] for _ in range(21)]


for col in range(allRain.shape[1]):
    imp_data_Rain[col].append(mean(allRain[:,col]))
    imp_data_Rain[col].append(std_dev(allRain[:,col]))

    imp_data_NotRain[col].append(mean(notRain[:,col]))
    imp_data_NotRain[col].append(std_dev(notRain[:,col]))
    

# find probabilities of each row now
def single_prob_Rain(data, col):
    m = imp_data_Rain[col][0]
    sd = imp_data_Rain[col][1]
    mult = math.exp(-((data - m)**2 / (2*sd**2)))
    return (1/(math.sqrt(2*math.pi)*sd)) * mult

def single_prob_NotRain(data, col):
    m = imp_data_NotRain[col][0]
    sd = imp_data_NotRain[col][1]
    mult = math.exp(-((data - m)**2 / (2*sd**2)))
    return (1/(math.sqrt(2*math.pi)*sd)) * mult

totalRains = allRain.shape[0]
totalNotRains = notRain.shape[0]
# Prior rain today given rain tomorrow
priorRT_RT = (list(allRain[:,20]).count(1) / totalRains)
# Prior rain today given not rain tomorrow
priorRT_NT = (list(notRain[:,20]).count(1) / totalNotRains)
# Prior not rain today given rain tomorrow
priorNT_RT = (list(allRain[:,20]).count(0) / totalRains)
# Prior not rain today given not rain tomorrow
priorNT_NT = (list(notRain[:,20]).count(0) / totalNotRains)

# Now for directions, hopefully do a more automated method...

priorsDirections_col6 = []
priorsDirections_col8 = []
priorsDirections_col9 = []


for direction in range(16):
    priorsDirections_col6.append(list(allRain[:,6]).count(direction) / totalRains)
    priorsDirections_col6.append(list(notRain[:,6]).count(direction) / totalNotRains)

    priorsDirections_col8.append(list(allRain[:,8]).count(direction) / totalRains)
    priorsDirections_col8.append(list(notRain[:,8]).count(direction) / totalNotRains)

    priorsDirections_col9.append(list(allRain[:,9]).count(direction) / totalRains)
    priorsDirections_col9.append(list(notRain[:,9]).count(direction) / totalNotRains)


# Using Bayes rule, find total probability that tomorrow will rain and not rain
pRain = allRain.shape[0] / traindata.shape[0]
pNotRain = notRain.shape[0] / traindata.shape[0]

combList1 = [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19]
probRain = np.ones(testdata.shape[0])
probNotRain = np.ones(testdata.shape[0])
decisions = []
arrayToUse = tuple(combList1) # Does not consider index 12 column
for row in range(testdata.shape[0]):
    for i in arrayToUse:
        if (i != 6 and i != 8 and i != 9):
            #if (single_prob_Rain(testdata[row][i],i) != 0):
            probRain[row] *= single_prob_Rain(testdata[row][i],i)
            #if (single_prob_NotRain(testdata[row][i],i) != 0):
            probNotRain[row] *= single_prob_NotRain(testdata[row][i],i)
        else:
            # For column 6 Directions:
            for val in range(16):
                if (testdata[row][6] == val):
                    if (priorsDirections_col6[2*val] != 0):
                        probRain[row] *= priorsDirections_col6[2*val]
                    if (priorsDirections_col6[2*val + 1] != 0):
                        probNotRain[row] *= priorsDirections_col6[2*val + 1]
        if (i == 8):
            #Column 8:
            for val in range(16):
                if (testdata[row][8] == val):
                    if (priorsDirections_col8[2*val] != 0):
                        probRain[row] *= priorsDirections_col8[2*val]
                    if (priorsDirections_col8[2*val+1] != 0):
                        probNotRain[row] *= priorsDirections_col8[2*val + 1]
        if (i == 9):
            # Column 9:
            for val in range(16):
                if (testdata[row][9] == val):
                    if (priorsDirections_col9[2*val] != 0):
                        probRain[row] *= priorsDirections_col9[2*val]
                    if (priorsDirections_col9[2*val+1] != 0):
                        probNotRain[row] *= priorsDirections_col9[2*val + 1]

    # For rainToday column
    if (testdata[row][20] == 1): # Rains today
        probRain[row] *= priorRT_RT
        probNotRain[row] *= priorRT_NT
    else:
        probRain[row] *= priorNT_RT
        probNotRain[row] *= priorNT_NT

    # Multiplying by priors
    probRain[row] *= -math.log(pRain,2)
    probNotRain[row] *= -math.log(pNotRain,2)
    if (probRain[row] > probNotRain[row]):
        decisions.append(1)
    else:
        decisions.append(0)

correct = 0
count = 0
for i in decisions:
    print(i)
    if (i == testdata[count][-1]):
        correct += 1
    count += 1
accuracy = correct/testdata.shape[0]

