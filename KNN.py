# Importing libraries
import numpy as np
import matplotlib.pyplot as plt

# Distance metric
def minkowski_distance(x1, x2, p):
    """
    Minkowski distance is a generalization of both manhattan distance (p=1)
    and euclidean distance (p=2)
    """
    return np.sum(np.abs(x1 - x2)**p)**(1/p)


class KNN:
    """
    This class implements K Nearest Neighbor Classifier
    Parameters
    =======================================================================
    k: number of nearest neighbors (default: 3)
    p: hyperparameter for minkowski distance (default: 2)
    weight: 
        uniform (default): All k nearest points assigned equal weights
        harmonic: Nearest instance of neighbor gets a weight of 1/1. second
                  nearest 1/2, the 1/3 and so on... 

    """
    def __init__(self, k = 3, p = 2, weight = 'uniform') -> None:
        self.k = k
        self.p = p
        self.weight = weight

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)

    def _vote(self, labels, weights):
        if self.weight == 'harmonic':
            pred = np.argmax(np.bincount(labels.astype(int), weights))

        else:
            pred = np.argmax(np.bincount(labels.astype('int')))
        return pred

    def _predict(self, x):
        distances = np.array([minkowski_distance(x, x_train, self.p) for x_train in self.X_train])
        idxs = np.argsort(distances)[:self.k]
        if self.weight == 'harmonic':
            w = np.array([1/i for i in range(1, len(idxs)+1) ])
        else:
            w = None
        k_nearest_labels = np.array([self.y_train[idx] for idx in idxs])
        prediction = self._vote(k_nearest_labels, w)
        return prediction 

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    #from sklearn.preprocessing import StandardScaler

    #scaler = StandardScaler()

    iris = datasets.load_iris()
    X, y = iris.data, iris.target


    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    k = 5
    p = 2
    weight = 'harmonic'
    classifier = KNN(k=k, p=p, weight=weight)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print("KNN classification accuracy is: ", accuracy(y_test, predictions))