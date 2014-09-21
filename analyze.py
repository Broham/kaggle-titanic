import csv as csv
import numpy as np

train = csv.reader(open('./kaggle/train.csv','rb'))
trainHeader = train.next()

test = csv.reader(open('./kaggle/test.csv','rb'))
testHeader = test.next()

prediction = csv.writer(open('genderPrediction.csv', 'wb'))

data = []
for row in train:
	data.append(row)
data = np.array(data)

numberPassengers = np.size(data[0::,1].astype(np.float))
numberSurvived = np.sum(data[0::,1].astype(np.float))
proportionSurvived = numberSurvived/numberPassengers

females = data[0::,4] == 'female' # creates list that is the same length as our data array.  Will have "true" at a particular index if the corresponding index record if for a female
males = data[0::,4] != 'female'
femaleSurvivorStats = data[females,1].astype(np.float) # selects the survived flag for each female record
maleSurvivorStats = data[males,1].astype(np.float)

proprotionFemaleSurvivors = np.sum(femaleSurvivorStats) / np.size(femaleSurvivorStats)
proprotionMaleSurvivors = np.sum(maleSurvivorStats) / np.size(maleSurvivorStats)

print "Percentage of female survivors: %s" % proprotionFemaleSurvivors
print "Percentage of male survivors: %s" % proprotionMaleSurvivors
# print maleStats
