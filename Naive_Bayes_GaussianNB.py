#naive bayes is calculated based on frequencies of the features so works well
#when you have discrete data with a small range of values shared across the data points
#good for classifying texts

from sklearn.naive_bayes import GaussianNB

#x is features while y is labels

#training data
x_train = [[1,2],[1,5],[5,3],[4,5],[7,3],[4,7]]
y_train = ['al', 'al', 'bl', 'al', 'bl', 'al']

#test data
x_test = [[4,5],[7,5],[2,5],[8,4]]
y_test = ['al', 'bl', 'al', 'bl']

#initialize classifies
clf = GaussianNB()

#fit classifies to training data
clf.fit(x_train, y_train)

print('Prediction of point [12,7]: ', clf.predict([[12,7]]))

#calculate accuracy
print('Automatically calculated accuracy: ', clf.score(x_test, y_test))

#manual algorithm to calculate accuracy
correct = 0
for n in range(len(x_test)):
    pred = clf.predict([x_test[n]])
    if pred==y_test[n]:
        correct+=1

accuracy = correct/len(y_test)
print('Manually calculated accuracy: ', accuracy)

