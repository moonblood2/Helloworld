from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

class myKN():

    def __init__(self,k=3):
        self.k=k
    
    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        prediction = []
        for row in x_test:
            label = self.closest(row)
            prediction.append(label)
        return prediction

    def closest(self, row):
        best_dist=np.linalg.norm(row-self.x_train[i])
        best_index=i
            
        for i in range(1,len(self.x_train)):
            dist = np.linalg.norm(row-self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[i]
        
iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

clf = myKN()
clf.fit(x_train,y_train)
print(accuracy_score(y_test,clf.predict(x_test)))
