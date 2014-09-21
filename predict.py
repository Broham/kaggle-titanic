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
#columns:
#0			 |1        |2      |3    |4   |5   |6     |7     |8      |9    |10    |11
#PassengerId |Survived |PClass |Name |Sex |Age |SibSp |Parch |Ticket |Fare |Cabin |Embarked


fareCeiling = 40

# if the fare column has a value greater than the fare ceiling, then set that value to fareCeiling - 1
data[ data[0::,9].astype(np.float) >= fareCeiling, 9 ] = fareCeiling - 1.0

fareBracketSize = 10
numberOfBrackets = fareCeiling / fareBracketSize

# find the number of unique classes
numberOfClasses = len(np.unique(data[0::,2])) 

# initialize a table with all zeros
survivalTable = np.zeros((2, numberOfClasses, numberOfBrackets))

for i in xrange(numberOfClasses): 
	for j in xrange(numberOfBrackets):

		#get all records where:
		#record is for a female and...
		#record is for the Ith class and...
		#record is greater than or equal than the Jth price bracket and...
		#record is less than the Jth price bracket and...
		womenStats = data[											\
			(data[0::,4] == 'female') & 							\
			(data[0::,2].astype(np.float) == i+1 ) &				\
			(data[0::,9].astype(np.float) >= j*fareBracketSize) &	\
			(data[0::,9].astype(np.float) < (j+1)*fareBracketSize)	\
			,1
		]

		#same as above but for male
		menStats = data[											\
			(data[0::,4] != 'female') & 							\
			(data[0::,2].astype(np.float) == i+1 ) &				\
			(data[0::,9].astype(np.float) >= j*fareBracketSize) &	\
			(data[0::,9].astype(np.float) < (j+1)*fareBracketSize)	\
			,1
		]

		survivalTable[0,i,j] = np.mean(womenStats.astype(np.float))
		survivalTable[1,i,j] = np.mean(menStats.astype(np.float))

#set nan's to 0
survivalTable[survivalTable != survivalTable] = 0		
survivalTable[survivalTable < .5] = 0
survivalTable[survivalTable >= .5] = 1

prediction.writerow(['PassengerId', 'Survived'])
for row in test:
	for j in xrange(numberOfBrackets):
		try:
			row[8] = float(row[8])
		except:
			binFare = 3 - float(row[1])
			break

		if row[8] > fareCeiling:
			binFare = numberOfBrackets - 1
			break
		if row[8] >= j * fareBracketSize and row[8] < (j+1) * fareBracketSize:
			binFare = j                           
			break  

	if row[3] == 'female':                             #If the passenger is female
		prediction.writerow([row[0], "%d" % int(survivalTable[0, float(row[1])-1, binFare])])
	else:                                          #else if male
		prediction.writerow([row[0], "%d" % int(survivalTable[1, float(row[1])-1, binFare])])
     
# Close out the files.
# test.close() 
# prediction.close()

print survivalTable












