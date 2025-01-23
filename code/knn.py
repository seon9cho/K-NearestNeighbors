import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import arff
import numpy.linalg as la
from sklearn.model_selection import train_test_split

def normalize(X, min_val=None, max_val=None):
    if min_val is None: min_val = np.nanmin(X, axis=0)
    if max_val is None: max_val = np.nanmax(X, axis=0)
    return (X - min_val) / (max_val - min_val)

class KNNClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self, k, output_type="nominal", input_type=0, weight_type='inverse_distance', normalize=True):
        """
        Args:
            k: number of nearest neighbors
            output_type: whether the problem is classification or regrssion
            input_type: if 0, the inputs are continuous; if 1, the inputs are nominal; if an array, then the inputs are both continuous and nominal depending on the value at each index.
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
            normalize: whehter to normalize the input dat aor not.
        """
        self.output_type = output_type
        self.input_type = input_type
        if type(input_type) == np.ndarray:
            self.num_noms = sum(input_type)
        self.weight_type = weight_type
        self.normalize = normalize
        self.k = k
        
    def fit(self, X, y):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        if type(self.input_type) == np.ndarray:
            self.X_nom = X.T[self.input_type==1].T
            X_cont = X.T[self.input_type==0].T
            if self.normalize:
                self.min_val = np.nanmin(X_cont, axis=0)
                self.max_val = np.nanmax(X_cont, axis=0)
                self.X_cont = normalize(X_cont)
        else:
            if self.normalize:
                self.min_val = np.nanmin(X, axis=0)
                self.max_val = np.nanmax(X, axis=0)
                self.X = normalize(X)
            else:
                self.X = X
        
        if self.output_type == "nominal":
            self.y = y.flatten().astype(int)
            self.num_classes = np.max(self.y) + 1
        else:
            self.y = y.flatten()
        return self
    
    def predict(self, X):
        if type(self.input_type) == np.ndarray and self.normalize:
            X_nom = X.T[self.input_type==1].T
            X_cont = X.T[self.input_type==0].T
            X_cont = normalize(X_cont, min_val=self.min_val, max_val=self.max_val)
            X = np.hstack([X_nom, X_cont])
        elif self.normalize:
            X = normalize(X, min_val=self.min_val, max_val=self.max_val)
        
        if self.output_type == "nominal":
            y_hat = self.predict_nominal(X)
        else:
            y_hat = self.predict_regress(X)
        return y_hat
    
    def topk(self, x):
        if type(self.input_type) == np.ndarray:
            x_nom = x[:self.num_noms]
            x_cont = x[self.num_noms:]
            nom_diff = np.sum(self.X_nom != x_nom, axis=1)
            cont_diff = self.X_cont - x_cont
            cont_diff[np.isnan(cont_diff)] = 1
            norms = np.sqrt(np.sum(cont_diff**2, axis=1) + nom_diff)
        else:
            norms = la.norm(self.X - x, axis=1)
        Ny = np.vstack([norms, self.y]).T
        sort_arg = np.argsort(norms)
        topk = Ny[sort_arg][:self.k]
        return topk
    
    def predict_nominal(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        y_hat = []
        for x in X:
            topk = self.topk(x)
            if self.weight_type == "no_weight":
                (values,counts) = np.unique(topk[:, -1].astype(int),return_counts=True)
                ind = np.argmax(counts)
                y_hat.append(values[ind])
            else:
                weights = topk[:, 0]**(-2)
                best_weight = 0
                best_ind = 0
                for i in range(self.num_classes):
                    curr_weight = np.sum(weights[topk[:, -1] == i])
                    if curr_weight > best_weight:
                        best_weight = curr_weight
                        best_ind = i
                y_hat.append(best_ind)
        return np.array(y_hat).reshape(-1, 1)
    
    def predict_regress(self, X):
        y_hat = []
        for x in X:
            topk = self.topk(x)
            if self.weight_type == "no_weight":
                y_hat.append(np.mean(topk[:, -1]))
            else:
                weights = topk[:, 0]**(-2)
                val = np.sum(weights*topk[:, -1])/np.sum(weights)
                y_hat.append(val)
        return np.array(y_hat).reshape(-1, 1)

    #Returns the Mean score given input data and labels
    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
                X (array-like): A 2D numpy array with data, excluding targets
                y (array-like): A 2D numpy array with targets
        Returns:
                score : float
                        Mean accuracy of self.predict(X) wrt. y.
        """
        y_hat = self.predict(X)
        if self.output_type == "nominal":
            return sum(y_hat==y) / len(y)
        else:
            return np.sum((y_hat - y)**2) / len(y)
            
from sklearn.base import clone

def greedy_wrapper(model, X_train, X_test, y_train, y_test, max_features=40):
    num_features = X_train.shape[1]
    best_features = []
    for i in range(max_features+1):
        scores = np.zeros(num_features)
        for j in range(num_features):
            if j in best_features:
                continue
            wrap = best_features + [j]
            W_train = X_train[:, wrap]
            W_test = X_test[:, wrap]
            clf = clone(model)
            clf.fit(W_train, y_train)
            score = clf.score(W_test, y_test)
            scores[j] = score
        best = np.argmax(scores)
        best_features.append(best)
        print(best_features)
        print(np.max(scores))
        print()