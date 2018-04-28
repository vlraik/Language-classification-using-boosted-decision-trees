from features import *
from classifiers import *
import pickle


def traininginput(input_file):
    """
    A function to read the training dataset from the file.

    :param input_file: The name of the input file.
    :return: The training dataset, along with the english and dutch dataset.
    """
    data = open(input_file, encoding="utf8").read().splitlines()
    englishdata, dutchdata = [], []
    maindata = []
    for i in data:
        maindata.append(i.lower())
        if i.startswith("nl|"):
            dutchdata.append(i[3:].lower())
        elif i.startswith("en|"):
            englishdata.append(i[3:].lower())
    return englishdata, dutchdata, data


def testinput(test_file):
    """
    A function to read the test dataset from the file.

    :param test_file: The name of the test file.
    :return: The test dataset as a list.
    """
    return open(test_file).read().splitlines()


def train(input_file, model_output, classifier):
    """
    Train the dataset based on the input parameters and also get the feautures from the training
    document corpus dataset.

    :param input_file: The input training file.
    :param model_output: The output file to store the model.
    :param classifier: The name of the classifier to use.
    """
    global english, dutch, tree, dt, ada

    english, dutch, data = traininginput(input_file)

    feature1 = np.asarray(lengthofwords(english, dutch, data))

    feature2 = np.asarray(freqoflettersinsentence(english, dutch, data))

    feature3 = np.asarray(uncommontopletters(english, dutch, data))

    feature4 = np.asarray(worduniqueness(english, dutch, data))

    feature5 = np.asarray(tfidf(english, dutch, data))

    feature6 = np.asarray(uniquewordsinsentence(english, dutch, data))

    feature7 = np.asarray(uniquelettersinasentence(english, dutch, data))

    feature8 = np.asarray(bigram(english, dutch, data))

    feature9 = np.asarray(trigram(english, dutch, data))

    feature10 = np.asarray(wordswithrepeatingletters(english, dutch, data))

    X_train = np.column_stack(
        (feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10))

    y = []
    for i in data:
        if i[:3] == 'nl|':
            y.append(0)
        elif i[:3] == 'en|':
            y.append(1)

    y = np.asarray(y)
    X_train = np.column_stack((X_train, y))

    if classifier == 'dt':
        tree = DecisionTree(X_train, y, maxdepth=5)
        dt = tree.build_tree(X_train)
        tree.rootnode(dt)
        file = open(model_output+'.obj','wb')
        pickle.dump(tree,file)


    elif classifier == 'ada':
        ada = Adaboost(X_train, y, 2)
        #print(type(ada))
        file = open(model_output+'.obj','wb')
        pickle.dump(ada,file)
        file.close()
        ada.train(X_train)
        #ada.classify(X_train)
        #a=pickle.load(open('out.obj','rb'))
        #print(type(a) is Adaboost)
    # savemodel(model_output)


def predict(model_name, test_file):
    """
    A function to classify the test dataset.

    :param model_name: The file name to load the model from
    :param test_file: The test file dataset.
    """
    # data = testinput(test_file)
    # loadmodel(model_name)
    data = testinput(model_name)
    model = pickle.load(open(test_file+'.obj','rb'))

    feature1 = np.asarray(lengthofwords(english, dutch, data))

    feature2 = np.asarray(freqoflettersinsentence(english, dutch, data))

    feature3 = np.asarray(uncommontopletters(english, dutch, data))

    feature4 = np.asarray(worduniqueness(english, dutch, data))

    feature5 = np.asarray(tfidf(english, dutch, data))

    feature6 = np.asarray(uniquewordsinsentence(english, dutch, data))

    feature7 = np.asarray(uniquelettersinasentence(english, dutch, data))

    feature8 = np.asarray(bigram(english, dutch, data))

    feature9 = np.asarray(trigram(english, dutch, data))

    feature10 = np.asarray(wordswithrepeatingletters(english, dutch, data))

    X_train = np.column_stack(
        (feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10))


    if type(model) is Adaboost:
        ada.classify(X_train)
    elif type(model) is DecisionTree:
        for i in X_train:
            # print(tree.classify(i,dt))
            answer = model.classify(i, model.dt)
            if 0 in answer:
                print("nl")
            elif 1 in answer:
                print("en")

