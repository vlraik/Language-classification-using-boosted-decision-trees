import numpy as np


class Node:
    def __init__(self, question, englishbranch, dutchbranch):
        """
        Node intialisation of the decision tree.

        :param question: Stores the node information.
        :param englishbranch: Datasets splitting into the English Branch
        :param dutchbranch: Datasets splitting into the Dutch Branch
        """
        self.question = question
        self.englishbranch = englishbranch
        self.dutchbranch = dutchbranch


class Leaf:
    def __init__(self, d, x):
        """
        Leaf node initialisation.

        :param d: Decision tree object
        :param x: Prediction
        """
        self.pred = DecisionTree.classcount(d, x)


class DecisionTree:
    def __init__(self, X_train, y, maxdepth=5):
        """
        Decision tree object intialisation.

        :param X_train: Dataset used for training
        :param y: The target value for the training dateset.
        :param maxdepth: The maximum depth of the decision tree
        """
        self.X_train = X_train[:, -1]
        self.y = y
        self.totaldutch = 0
        self.totalenglish = 0
        self.maxdepth = maxdepth
        self.dt = None
        # self.data = np.column_stack((self.X_train, self.y))

    def rootnode(self, dt):
        """
        A function to declare the rootnode the the decision tree attribute.

        :param dt: The root node
        """
        self.dt = dt

    def build_tree(self, x, depth=5):
        """
        A function to build the tree based on the training set.

        :param x: Training set
        :param depth: Depth of the recursion
        :return: The parent node at that level of the decision tree.
        """
        gain, question = self.find_best_split(x)
        # print(question.val)
        # print(question.col)
        # print(question)
        if gain != 0:
            englishrows = []
            dutchrows = []
            for k in x:
                if question.match(k) == False:
                    dutchrows.append(k)
                else:
                    englishrows.append(k)
            englishbranch, dutchbranch = np.asarray(englishrows), np.asarray(dutchrows)
            # englishbranch, dutchbranch = self.partition(x, question)
            # print(englishbranch)
            # print(dutchbranch)

            if depth <= self.maxdepth:
                depth -= 1
                englishbranch = self.build_tree(englishbranch, depth)
                dutchbranch = self.build_tree(dutchbranch, depth)


        elif gain == 0:
            return Leaf(self, x)

        return Node(question, englishbranch, dutchbranch)

    def classify(self, x, node):
        """
        A function for classifying the data based on the decision tree.

        :param x: Testing example.
        :param node: The root node of the decision tree
        :return: The prediction.
        """
        if isinstance(node, Leaf):
            return node.pred

        if node.question.match(x):
            return self.classify(x, node.englishbranch)
        else:
            return self.classify(x, node.dutchbranch)

    def classcount(self, x):
        """
        A function to get the count of the different labels in the dataset

        :param x: The branch of the dataset in which we want to count.
        :return: Number of counts.
        """
        counts = {}
        for i in range(len(x)):
            if x[i, -1] in counts:
                counts[x[i, -1]] += 1
            else:
                counts[x[i, -1]] = 1

        return counts

    def gini(self, x):
        """
        A function to get the gini impurity of the dataset.

        :param x: The dataset
        :return: Gini Impurity
        """
        counts = self.classcount(x)
        impurity = 1
        for i in counts:
            prob = counts[i] / float(len(x))
            impurity -= prob ** 2
        return impurity

    def info_gain(self, startnode, left, right):
        """
        A function to get the information gain at each node of the decision tree.

        :param startnode: The node at which we want to get the information gain.
        :param left: The left branch from the node.
        :param right: The right branch from the node.
        :return: The information gain at that node.
        """
        p = float(len(left) / (len(right) + len(left)))
        gain = self.gini(startnode) - p * self.gini(left) - (1 - p) * self.gini(right)
        return gain

    def find_best_split(self, x):
        """
        A function to find the based node based on the gini impurity and information gain of node.

        :param x: The training dataset.
        :return: The gain and the node which we find best divides the dataset.
        """
        gain, question = 0, None
        for i in range(10):
            values = [0, 1]
            for j in values:
                # print(i,j)
                currentquestion = PartitionMatch(i, j)
                englishrows = []
                dutchrows = []
                for k in x:
                    if currentquestion.match(k) == False:
                        dutchrows.append(k)
                    else:
                        englishrows.append(k)
                englishsplit, dutchsplit = np.asarray(englishrows), np.asarray(dutchrows)
                if len(englishsplit) == 0 or len(dutchsplit) == 0:
                    continue
                currentgain = self.info_gain(x, englishsplit, dutchsplit)
                # print()
                if currentgain < gain:
                    continue
                else:
                    gain = currentgain
                    question = currentquestion

        return gain, question


class PartitionMatch:

    def __init__(self, col, val):
        """
        Class initialisation to store the structure of the tree.

        :param col: The feature which is at the node.
        :param val: The value which it predicts.
        """
        self.col = col
        self.val = val

    def match(self, sample):
        """
        A function to get what a particular node predicts.

        :param sample: The value of the feature.
        :return: Binary, the prediction.
        """
        return sample[self.col] == self.val


class Adaboost():

    def __init__(self, X_train, y, n_trees=2):
        """
        Initialisation of the adaboost class.

        :param X_train: The training dataset.
        :param y: The target value
        :param n_trees: Number of decision stump inside the adaboost.
        """
        self.n_trees = n_trees
        self.y = y
        self.X_train = X_train

    def train(self, X_train):
        """
        A function to train the dataset on adaboost.

        :param X_train: The training dataset.

        """
        exampleweight = [1 / len(X_train)] * len(X_train)
        modelweight = [0.5] * self.n_trees

        models = [None] * self.n_trees
        dt = [None] * self.n_trees
        for epoch in range(20):
            for i in range(self.n_trees):

                randomsamplesindex = [i for i in range(len(X_train))]

                index = np.random.choice(randomsamplesindex, len(X_train) // self.n_trees, p=exampleweight)

                randomsamples = [X_train[i] for i in index]
                randomsamples = np.asarray(randomsamples)
                models[i] = DecisionTree(randomsamples, randomsamples[:, -1], maxdepth=5)
                dt[i] = models[i].build_tree(randomsamples)

                answers = []
                for j in X_train:
                    if 0 in models[i].classify(j, dt[i]):
                        answers.append("nl|")
                    elif 1 in models[i].classify(j, dt[i]):
                        answers.append("en|")

                accuracy = 0
                for j in range(len(answers)):
                    if answers[j] == self.y[j]:
                        accuracy += 1
                        # exampleweight[j]-=exampleweight[j]/2
                    elif answers[j] != self.y[i]:
                        # exampleweight[j]=1/(len(X_train)-0.2*len(X_train))
                        pass

                for j in range(len(answers)):
                    if accuracy != 0:
                        for j in range(len(answers)):
                            if answers[j] == self.y[j]:
                                exampleweight[j] = 1 / (accuracy / 0.4)
                            elif answers[j] != self.y[j]:
                                exampleweight[j] = 1 / ((len(X_train) - accuracy) / 0.6)

                accuracy = accuracy / len(answers)

                if accuracy == 0.5:
                    modelweight[i] = 0
                elif accuracy > 0.5:
                    modelweight[i] = 1
                elif accuracy < 0.5:
                    modelweight[i] = -1

        self.modelweight = modelweight
        self.models = models
        self.dt = dt

    def classify(self, X_train):
        """
        A function to classify a test example based on the adaboost training.

        :param X_train: The test example dataset.

        """
        modelweight = self.modelweight
        models = self.models
        dt = self.dt
        finalresults = [None] * self.n_trees
        for i in range(self.n_trees):
            answers = []
            for j in X_train:
                answers.append(models[i].classify(j, dt[i]))
            finalresults[i] = answers

        averageresult = [0] * len(self.y)
        for j in range(len(self.y)):
            for i in range(len(finalresults)):
                for k in finalresults[i][j]:
                    if 0 == k:
                        averageresult[j] += modelweight[i] * 0
                    elif 1 == k:
                        averageresult[j] += modelweight[i] * 1
            averageresult[j] = averageresult[j] / sum(modelweight)

        for i in averageresult:
            if i > 0.5:
                print("en")
            else:
                print("nl")
