
# coding: utf-8

# # C4.5
# 
# To improve on the shortcomings of ID3, we'll implement C4.5 and its main ideas:
# 
# - Splitting continuous features smarter
# - Handling missing values directly
# - Growing the tree as far as possible and pruning it later

# ## Helper functions
# 
# A few general functions are needed for C4.5. We'll define them here because they might come in handy in another situation.

# In[1]:

def mean(xs):
    return float(sum(xs)) / len(xs)


# In[2]:

import math
import numpy as np

def isnan(val):
     return type(val) == float and math.isnan(val)


# We'll also want to use this function on numpy arrays because `np.isnan` doesn't work for some datatypes.

# In[3]:

isnan = np.vectorize(isnan)


# ## C4.5 implementation
# 
# Generally, this is based on the ID3 implementation, but a lot of new stuff is added.

# In[4]:

import numpy as np
from collections import defaultdict, Counter
from math import log as logarithm
from operator import itemgetter
import Queue
    
class C45:
    def __init__(self, max_depth=float("inf"), min_gain=0, continuous={}, depth=0):
        """
        Arguments:
            max_depth: After eaching this depth, the current node is turned into a leaf which predicts
                the most common label. This limits the capacity of the classifier and helps combat overfitting
            min_gain: The minimum gain a split has to yield. Again, this helps overfitting
            depth: Let's the current node know how deep it is into the tree, users usually don't need to set this
        """
        
        self.depth = depth
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.continuous = continuous
        
        # ID3 nodes are either nodes that make a decision or leafs which constantly predict the same result
        # We represent both possibilities using `ID3` objects and set `self.leaf` respectively
        self.leaf = False
        self.value = None
        
        self.children = {}
        self.feature = 0
        self.feature_split = None
    
    def fit(self, X, y):
        """
        Creates a tree structure based on the passed data
        
        Arguments:
            X: numpy array that contains the features in its rows
            y: numpy array that contains the respective labels
        """
        
        self.counts = Counter(y)
        self.most_common_label = self.counts.most_common()[0][0]
        
        # If there is only one class left, turn this node into a leaf
        # and always return this one value
        if len(set(y)) == 1:
            self.leaf = True
            self.value = y[0]
        # If the tree is getting to deep, turn this node into a leaf
        # and always predict the most common value
        elif self.depth >= self.max_depth:
            self.leaf = True
            self.value = self.most_common_label
        elif len({tuple(row) for row in X}) == 1:
            self.leaf = True
            self.value = self.most_common_label
        # Otherwise, look for the most informative feature and do a split on its possible values
        else:
            self.feature, self.feature_split = self._choose_feature(X, y)
            
            # If no feature is informative enough, turn this node into a leaf
            # and always predict the most common value
            if self.feature is None:
                self.leaf = True
                self.value = self.most_common_label
            else:
                if self.feature in self.continuous:
                    partition = self._partition_continuous(X, y, self.feature, self.feature_split)
                else:
                    partition = self._partition(X, y, self.feature)
                    
                if self._is_useful_partition(partition):
                    self._save_partition_proportions(partition)
                    
                    for value, (Xi, yi) in partition.iteritems():
                        child = C45(continuous=self.continuous, depth=self.depth+1, max_depth=self.max_depth)
                        child.fit(Xi, yi)
                        self.children[value] = child
                else:
                    self.leaf = True
                    self.value = self.most_common_label
    
    def predict_single(self, x):
        """
        Predict the class of a single data point x by either using the value encoded in a leaf
        or by following the tree structure recursively until a leaf is reached
        
        Arguments:
            x: individual data point
        """
        
        if self.leaf:
            return self.value
        else:
            value = x[self.feature]
            
            if isnan(value):
                return self._get_random_child_node().predict_single(x)
            elif self.feature in self.continuous:
                return self._predict_single_continuous(x, value)
            else:
                return self._predict_single_discrete(x, value)
                
    def _predict_single_discrete(self, x, value):
        if value in self.children:
            return self.children[value].predict_single(x)
        else:
            return self.most_common_label
        
    def _predict_single_continuous(self, x, value):
        if value <= self.feature_split:
            node = "smaller"
        else:
            node = "greater"

        return self.children[node].predict_single(x)
        
    def predict(self, X):
        """
        Predict the results for an entire dataset
        
        Arguments:
            X: numpy array that contains each data point in a row
        """
        
        return [self.predict_single(x) for x in X]
    
    def score(self, X, y):
        """
        Returns the accuracy for predicting the given dataset X
        """
        
        correct = sum(self.predict(X) == y)
        return float(correct) / len(y)
        
    def _choose_feature(self, X, y):
        """
        Finds the most informative feature to split on and returns its index.
        If no feature is informative enough, `None` is returned
        """
        
        best_feature = 0
        best_feature_gain = -float("inf")
        best_feature_split = None
        
        for i in range(X.shape[1]):
            gain, split = self._information_gain(X, y, i)

            if gain > best_feature_gain:
                best_feature = i
                best_feature_gain = gain
                best_feature_split = split
                        
        if best_feature_gain < self.min_gain:
            best_feature = None
            
        self.gain = best_feature_gain
            
        return best_feature, best_feature_split
        
    def _information_gain(self, X, y, feature):
        if feature in self.continuous:
            max_gain, best_split = self._information_gain_continuous(X, y, feature)
            return max_gain, best_split
        else:
            return self._information_gain_discrete(X, y, feature), 0
    
    def _information_gain_continuous(self, X, y, feature):
        """
        Calculates the information gain achieved by splitting on the given feature
        """
        
        data, splits = self._get_continuous_splits(X, y, feature)
        
        old_entropy = self._entropy(y)
        
        max_gain = -float("inf")
        best_split = None
        
        for split in splits:
            smaller = [yi for (xi, yi) in data if xi <= split]
            greater = [yi for (xi, yi) in data if xi > split]
                        
            ratio_smaller = float(len(smaller)) / len(data)
            
            new_entropy = ratio_smaller * self._entropy(smaller) + (1 - ratio_smaller) * self._entropy(greater)
            
            result = old_entropy - new_entropy
            
            if result > max_gain:
                best_split = split
                max_gain = result
        
        return max_gain, best_split
    
    def _information_gain_discrete(self, X, y, feature):
        """
        Calculates the information gain achieved by splitting on the given feature
        """
        
        result = self._entropy(y)
        
        summed = 0
        
        for value, (Xi, yi) in self._partition(X, y, feature).iteritems():
            # Missing values should be ignored for computing the entropy
            if not isnan(value):
                summed += float(len(yi)) / len(y) * self._entropy(yi)
        
        result -= summed
        
        return result
    
    def _entropy(self, X):
        """
        Calculates the Shannon entropy on the given data X
        
        Arguments:
            X: An iterable for feature values. Usually, this is now a 1D list
        """
        
        summed = 0
        counter = Counter(X)

        for value in counter:
            count = counter[value]
            px = count / float(len(X))
            summed += px * logarithm(1. / px, 2)
        
        return summed
    
    def _partition(self, X, y, feature):
        """
        Partitioning is a common operation needed for decision trees (or search trees).
        Here, a partitioning is represented by a dictionary. The keys are values that the feature
        can take. Under each key, we save a tuple (Xi, yi) that represents all data points (and their labels)
        that have the respective value in the specified feature.
        """
        
        partition = defaultdict(lambda: ([], []))
        
        for Xi, yi in zip(X, y):
            bucket = Xi[feature]
            
            partition[bucket][0].append(Xi)
            partition[bucket][1].append(yi)
        
        partition = dict(partition)
            
        for feature, (Xi, yi) in partition.iteritems():
            partition[feature] = (np.array(Xi), np.array(yi))
            
        return partition
    
    def _partition_continuous(self, X, y, feature, split):
        xi = X[:, feature]
        smaller = xi <= split
        greater = xi > split
        unknown = isnan(xi)
        
        ratio_smaller = sum(smaller) / float(sum(smaller) + sum(greater))
        
        unknown = np.where(unknown)[0]
        np.random.shuffle(unknown)
                
        #num_first = int(ratio_smaller * len(unknown))
        #smaller[unknown[:num_first]] = True
        #greater[unknown[num_first:]] = True
                
        greater[unknown] = True
        #smaller[unknown[:len(unknown)/2]] = True
        #greater[unknown[len(unknown)/2:]] = True
        
        num_smaller = sum(smaller)
        num_greater = sum(greater)
        
        #for i in unknown:
        #    if num_smaller < num_greater:
        #        smaller[i] = True
        #        num_smaller += 1
        #    else:
        #        greater[i] = True
        #        num_greater += 1
                        
        partition = {
            "smaller": (X[smaller], y[smaller]),
            "greater": (X[greater], y[greater])
        }
        
        return partition
    
    def _get_continuous_splits(self, X, y, feature):
        yi = y
        xi = X[:, feature]
        
        datai = sorted(zip(xi, yi), key=itemgetter(0, 1))

        splits = []

        xs = []
        ys = []
        last_x = None

        for xj, yj in datai:
            # Missing values can't be used to find good thresholds
            if isnan(xj):
                continue
                
            if xj == last_x:
                xs[-1].append(xj)
                ys[-1].add(yj)
            else:
                xs.append([xj])
                ys.append({yj})

            last_x = xj

        last_label = None

        for xj, yj in zip(xs, ys):
            if len(yj) == 1 and list(yj)[0] == last_label:
                splits[-1] += xj
            else:
                splits.append(xj)

            if len(yj) == 1:
                last_label = list(yj)[0]
            else:
                last_label = None

        splits = [mean(vals) for vals in splits]
        
        return datai, splits
    
    def _is_useful_partition(self, partition):
        num_useful = 0
        
        for value, (Xi, yi) in partition.iteritems():
            if len(yi) > 0:
                num_useful += 1
                
        return num_useful >= 2
    
    def _save_partition_proportions(self, partition):
        occurences = {}
        
        for child, (xj, xi) in partition.iteritems():
            occurences[child] = len(xj)
            
        total = float(sum(occurences.values()))
            
        self.children_probs = { child: occ / total for child, occ in occurences.iteritems() }
    
    def _get_random_child_node(self):
        name = np.random.choice(self.children_probs.keys(), p=self.children_probs.values())
        return self.children[name]
    
    def prune(self, X_val, y_val):
        old_score = self.score(X_val, y_val)
        pruned = 0
        not_pruned = 0
        
        for node in reversed(self._bfs()):
            if not node.leaf:
                node._make_leaf()
                score = self.score(X_val, y_val)
                
                if old_score > score:
                    node._make_internal()
                    not_pruned += 1
                else:
                    old_score = score
                    pruned += 1
                    
        return not_pruned, pruned
    
    def _bfs(self):
        queue = Queue.Queue()
        node = self
        
        queue.put(node)
        result = [node]
        
        while not queue.empty():
            if not node.leaf:
                for _, child in node.children.iteritems():
                    queue.put(child)
                    
            node = queue.get()
            result.append(node)
            
        return result
    
    def _make_leaf(self):
        self.leaf = True
        self.value = self.most_common_label
        
    def _make_internal(self):
        self.leaf = False
        self.value = None


# ## Titanic dataset
# 
# This is the same dataset as in the ID3 notebook, but now we can use continuous features, too.
# 
# The head of the dataset is always printed, to show how the data we are working on looks like.
# Depending on the exact C4.5 functionality being used, we might want to encode labels or impute values.

# In[5]:

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.model_selection import train_test_split


# In[6]:

def preprocess(data, encode_labels=False, impute=False):
    X = data.drop(["Survived", "Name", "Ticket", "Cabin"], 1)    
    
    if encode_labels: # for sklearn
        X = X.apply(LabelEncoder().fit_transform)
    
    print X.head(10)
    
    X = X.as_matrix()
    
    if impute:
        X = Imputer().fit_transform(X)
            
    return X


# In[7]:

data = DataFrame.from_csv("./titanic/train.csv")
y = data["Survived"].as_matrix()


# ### Continuous features
# 
# Missing values are imputed, and labels are encoded numerically. In contrast to ID3, we can now use continuous variables like age or the price paid. We need to tell C4.5 which variables are continuous, because this is hard (or error-prone) to automatically derive.

# In[8]:

X = preprocess(data, encode_labels=True, impute=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:

clf = C45(continuous={2, 5})
clf.fit(X_train, y_train)

print "train accuracy = %.5f" % clf.score(X_train, y_train)
print "test accuracy = %.5f" % clf.score(X_test, y_test)


# As we can see, training accuracy is way up compared to ID3, but test accuracy actually got worse. This is because it's now much easier to overfit.

# ### Missing values
# 
# As explained in the report, one major feature of C4.5 is that we don't have to manually impute missing values. So now, let's set the preprocessor up to not impute, and see how C4.5 deals with that.

# In[10]:

X = preprocess(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:

clf = C45(continuous={2, 5})
clf.fit(X_train, y_train)

print "train accuracy = %.5f" % clf.score(X_train, y_train)
print "test accuracy = %.5f" % clf.score(X_test, y_test)


# In[12]:

mean([clf.score(X_test, y_test) for _ in range(100)])


# As we can see, training accuracy got down. This is because it's harder to describe data with missing values. On the other hand, test accuracy is now improved, which is often our main goal.
# 
# Because there's now some randomness involved, we compute the mean of many test scores.

# ### Pruning
# 
# Next, we will grow the tree as far as possible and then prune the parts that don't help.
# 
# We will split our training set into a dataset that is used directly for training, and a validation set that is used to evaluate the nodes in the tree.

# In[13]:

X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size = 0.2)


# In[14]:

clf = C45(continuous={2, 5})
clf.fit(X_train_sub, y_train_sub)
clf.prune(X_val, y_val)


# The return value tells us that 23 nodes were pruned and 314 nodes were kept. This node count excludes existing leafs.

# In[15]:

print "train accuracy = %.5f" % clf.score(X_train, y_train)
print "test accuracy = %.5f" % clf.score(X_test, y_test)


# In[16]:

mean([clf.score(X_test, y_test) for _ in range(100)])


# As we can see, this improves the accuracy a bit more. However, we are still below the accuracy we get using the simple ID3 model, where we do some prior feature selection (i.e. exclude the continuous features).

# In[17]:

clf = C45(continuous={2, 5})
clf.fit(X_train, y_train)
clf.prune(X_test, y_test)

print "train accuracy = %.5f" % clf.score(X_train, y_train)
print "test accuracy = %.5f" % clf.score(X_test, y_test)


# Of course, when using the test set for pruning, we get much improved test accuracy. However, this is cheating because we shouldn't use test data for model selection.

# ## Drawing
# 
# Like in the ID3 notebook, we want to draw our results. We made some modifications to the draw functions, to draw continuous splits in a nice way.

# In[18]:

def setNodeId(depth,index=0):
    return str(int(depth)) + str(int(index))


# In[19]:

def show_content(node, result_list):
    i = 0
    node_txt = ''
    while i < len(node.counts.keys()):
        tmp_result = ''
        
        number = node.counts[node.counts.keys()[i]]                        
        tmp_result = result_list[node.counts.keys()[i]] + ': ' + str(number) + '\n'
        
        node_txt += tmp_result
                        
        i += 1
    return node_txt


# In[20]:

import Queue
import pydot



def draw(node,feature_list, result_list, path):
    
    graph = pydot.Dot(graph_type='graph')
    
    cid = 0
    
    que = Queue.Queue()
    
    node.Id = setNodeId(node.depth)
    que.put(node)
    
    while(que.qsize() > 0):
        
        node = que.get()
        
        feature = feature_list[node.feature]
        
        node_txt =  feature + '\n' + show_content(node, result_list)
        
        graph.add_node(pydot.Node(node.Id, label = node_txt))
        
        
        for index in node.children.keys():
            if node.children[index].leaf == True:
                if len(node.children[index].counts.keys()) == 1:
                    edge_txt = ''
                    
                    node.children[index].Id = setNodeId(node.children[index].depth, cid)                
                    
                    value = node.children[index].counts[node.children[index].counts.keys()[0]]
                    result = result_list[node.children[index].counts.keys()[0]]
                                        
                    graph.add_node(pydot.Node(node.children[index].Id, label = result + "\n" + str(value), shape = 'box'))
                    
                    if node.feature not in node.continuous:
                        edge_txt = str(index)
                    else:
                        if str(index) == "smaller":
                            edge_txt = u'≤' + str(node.feature_split)
                        else:
                            edge_txt = '>' + str(node.feature_split)
                    
                    edge = pydot.Edge(node.Id, node.children[index].Id, label= edge_txt)
                    graph.add_edge(edge)
                    
                    cid += 1
                else:
                    edge_txt = ''
                    node_txt = show_content(node.children[index], result_list)
                    
                    node.children[index].Id = setNodeId(node.children[index].depth, cid)
                    graph.add_node(pydot.Node(node.children[index].Id, label = node_txt, shape = 'box'))
                    
                    if node.feature not in node.continuous:
                        edge_txt = str(index)
                    else:
                        if str(index) == "smaller":
                            edge_txt = u'≤' + str(node.feature_split)
                        else:
                            edge_txt = '>' + str(node.feature_split)
                    
                    edge = pydot.Edge(node.Id, node.children[index].Id, label= edge_txt)
                    graph.add_edge(edge)
                    
                    cid += 1
                    
            else:
                edge_txt = ''
                node.children[index].Id = setNodeId(node.children[index].depth, cid)                
                
                if node.feature not in node.continuous:
                    edge_txt = str(index)
                else:
                    if str(index) == "smaller":
                        edge_txt = u'≤' + str(node.feature_split)
                    else:
                        edge_txt = '>' + str(node.feature_split)
                
                edge = pydot.Edge(node.Id, node.children[index].Id, label = edge_txt)
                graph.add_edge(edge)
                que.put(node.children[index])
                cid += 1
    
    graph.write_png(path)


# In[25]:

clf = C45(continuous={2, 5}, max_depth=2)
clf.fit(X_train, y_train)

feature_list = ['Pclass',  'Sex',  'Age',  'SibSp',  'Parch', 'Fare',  'Embarked']
survive_list = ['Not Survived', 'Survived']
draw(clf, feature_list, survive_list, path="c45-depth2.png")


# In[26]:

clf = C45(continuous={2, 5}, max_depth=3)
clf.fit(X_train, y_train)

feature_list = ['Pclass',  'Sex',  'Age',  'SibSp',  'Parch', 'Fare',  'Embarked']
survive_list = ['Not Survived', 'Survived']
draw(clf, feature_list, survive_list, path="c45-depth3.png")


# In[27]:

clf = C45(continuous={2, 5}, max_depth=4)
clf.fit(X_train, y_train)

feature_list = ['Pclass',  'Sex',  'Age',  'SibSp',  'Parch', 'Fare',  'Embarked']
survive_list = ['Not Survived', 'Survived']
draw(clf, feature_list, survive_list, path="c45-depth4.png")


# ## Checking with sklearn
# 
# Again, we can take a look at how well sklearn does on this dataset to see if we're totally off with our results.
# sklearn implements the CART algorithm for decision trees. Additionally, this implementation cannot deal with missing values, so we need to impute them and encode all labels numerically.

# In[51]:

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[52]:

X = preprocess(data, encode_labels=True, impute=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[60]:

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print "train accuracy = %.5f" % clf.score(X_train, y_train)
print "test accuracy = %.5f" % clf.score(X_test, y_test)


# As we can see, sklearn performs very similiarly in terms of test accuracy. We got a mean of `0.8058100558659214` using C4.5 without imputed values, which is just a little bit better than sklearn's performance:

# In[61]:

mean([clf.score(X_test, y_test) for _ in range(100)])

