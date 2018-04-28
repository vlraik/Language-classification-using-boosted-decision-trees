from collections import Counter


def lengthofwords(english, dutch, data):
    """
    A function to calculate the average length of words in each of the language.

    :param english: English language dataset
    :param dutch: Dutch language dataset.
    :param data: The entire training data
    :return: The feature for each of the samples.
    """
    feature1 = []
    englishaverage = 0
    dutchaverage = 0
    for i in english:
        for j in i.split():
            englishaverage += len(j)

    englishaverage = englishaverage / (15 * len(english))

    for i in dutch:
        for j in i.split():
            dutchaverage += len(j)

    dutchaverage = dutchaverage / (15 * len(dutch))

    threshold = (englishaverage + dutchaverage) / 2

    for i in data:
        i = i[3:]
        total = 0
        for j in i.split():
            total += len(j)
        total = total / 15

        if total > threshold:
            feature1.append(0)
        else:
            feature1.append(1)

    # returns a boolean feature column based on length of words feature.
    return feature1


def freqoflettersinsentence(english, dutch, data):
    """
    A feature based on the frequency of letters in each of the languages

    :param english: English language dataset
    :param dutch: Dutch language dataset.
    :param data: The entire training data
    :return: The feature for each of the samples.
    """
    feature = []

    english = ' '.join(english)
    dutch = ' '.join(dutch)
    englishcounter = Counter(english)
    englishcounter.pop(' ')
    dutchcounter = Counter(dutch)
    dutchcounter.pop(' ')
    englishfeatures = []
    dutchfeatures = []
    for i in range(10):
        englishfeatures.append(max(englishcounter, key=englishcounter.get))
        englishcounter.pop(max(englishcounter, key=englishcounter.get))
        dutchfeatures.append(max(dutchcounter, key=dutchcounter.get))
        dutchcounter.pop(max(dutchcounter, key=dutchcounter.get))

    for i in data:
        i = i[3:]
        datacounter = Counter(i)
        datacounter.pop(' ')
        datafeatures = []
        for j in range(10):
            datafeatures.append(max(datacounter, key=datacounter.get))
            datacounter.pop(max(datacounter, key=datacounter.get))
        englishfault = 0
        dutchfault = 0

        for a, b, c in zip(datafeatures, englishfeatures, dutchfeatures):
            if a != b:
                englishfault += 1
            if a != c:
                dutchfault += 1

        if englishfault > dutchfault:
            feature.append(0)
        else:
            feature.append(1)

    return feature


def uncommontopletters(english, dutch, data):
    """
    A feature based on the most infrequent common letters in both the language. If more of those language
    exist in a particular language corpes, we can predict that it's a sentence from that language.

    :param english: English language dataset
    :param dutch: Dutch language dataset.
    :param data: The entire training data
    :return: The feature for each of the samples.
    """
    feature = []

    english = ' '.join(english)
    dutch = ' '.join(dutch)
    englishcounter = Counter(english)
    englishcounter.pop(' ')
    dutchcounter = Counter(dutch)
    dutchcounter.pop(' ')
    englishfeatures = []
    dutchfeatures = []

    for i in range(10):
        englishfeatures.append(max(englishcounter, key=englishcounter.get))
        englishcounter.pop(max(englishcounter, key=englishcounter.get))
        dutchfeatures.append(max(dutchcounter, key=dutchcounter.get))
        dutchcounter.pop(max(dutchcounter, key=dutchcounter.get))

    englishset = set(englishfeatures) - set(dutchfeatures)
    dutchset = set(dutchfeatures) - set(englishfeatures)

    for i in data:
        i = i[3:]
        datacounter = Counter(i)
        datacounter.pop(' ')
        datafeatures = []
        for j in range(10):
            datafeatures.append(max(datacounter, key=datacounter.get))
            datacounter.pop(max(datacounter, key=datacounter.get))

        englishscore = 0
        dutchscore = 0
        for j in datafeatures:
            if j in englishset:
                englishscore += 1
            elif j in dutchset:
                dutchscore += 1

        if englishscore > dutchscore:
            feature.append(0)
        else:
            feature.append(1)

    return feature


def worduniqueness(english, dutch, data):
    """
    A feature to get how many unique letters are used apart from the commonly occuring letters of the language
    in the corpus.

    :param english: English language dataset
    :param dutch: Dutch language dataset.
    :param data: The entire training data
    :return: The feature for each of the samples.
    """
    feature = []

    english = ' '.join(english)
    dutch = ' '.join(dutch)
    englishcounter = Counter(english)
    englishcounter.pop(' ')
    dutchcounter = Counter(dutch)
    dutchcounter.pop(' ')
    englishfeatures = []
    dutchfeatures = []

    for i in range(10):
        englishfeatures.append(max(englishcounter, key=englishcounter.get))
        englishcounter.pop(max(englishcounter, key=englishcounter.get))
        dutchfeatures.append(max(dutchcounter, key=dutchcounter.get))
        dutchcounter.pop(max(dutchcounter, key=dutchcounter.get))

    for i in data:
        i = i[3:]
        englishuniqueletters = 0
        dutchuniqueletters = 0
        for j in i:
            letters = set(j)
            englishuniqueletters += len(letters - set(englishfeatures)) / len(letters)
            dutchuniqueletters += len(letters - set(dutchfeatures)) / len(letters)
        if englishuniqueletters > dutchuniqueletters:
            feature.append(0)
        else:
            feature.append(1)

    return feature


def tfidf(english, dutch, data):
    """
    A feature to get the most common words from each of the language and check if they exist in the document corpus.

    :param english: English language dataset
    :param dutch: Dutch language dataset.
    :param data: The entire training data
    :return: The feature for each of the samples.
    """
    feature = []
    englishconcat = ' '.join(english)
    dutchconcat = ' '.join(dutch)
    englishcommon = Counter(englishconcat.split()).most_common()
    dutchcommon = Counter(dutchconcat.split()).most_common()
    englishcommon = englishcommon[:3]
    dutchcommon = dutchcommon[:3]
    englishcommon = [i[0] for i in englishcommon]
    dutchcommon = [i[0] for i in dutchcommon]

    for i in data:
        englishscore = 0
        dutchscore = 0
        i = i[3:]
        datawords = i.split()
        for j in datawords:
            if j in englishcommon:
                englishscore += 1
            elif j in dutchcommon:
                dutchscore += 1
        if englishscore > dutchscore:
            feature.append(0)
        else:
            feature.append(1)
    return feature
    #return [0,0,0,0,0,0,0,0,0,0]


def uniquewordsinsentence(english, dutch, data):
    """
    A feature to count the number of unqiue words in a sentence on average.

    :param english: English language dataset
    :param dutch: Dutch language dataset.
    :param data: The entire training data
    :return: The feature for each of the samples.
    """
    feature = []
    englishaverage = 0
    for i in english:
        englishunique = set(i)
        englishaverage += len(englishunique)
    dutchaverage = 0
    for i in dutch:
        dutchunique = set(i)
        dutchaverage += len(dutchunique)

    englishaverage = englishaverage / len(english)
    dutchaverage = dutchaverage / len(dutch)
    threshold = (englishaverage + dutchaverage) / 2
    for i in data:
        i = i[3:]
        dataunique = set(i)
        dataaverage = len(dataunique)
        if dataaverage > threshold:
            feature.append(0)
        else:
            feature.append(1)

    return feature


def uniquelettersinasentence(english, dutch, data):
    """
    A feature to get the number of unique letters in the document corpus.

    :param english: English language dataset
    :param dutch: Dutch language dataset.
    :param data: The entire training data
    :return: The feature for each of the samples.
    """
    feature = []
    englishaverage = 0
    englishunique = set(list(' '.join(english)))
    englishunique.remove(' ')
    notalpha=[]
    for i in englishunique:
        if i.isalpha():
            continue
        else:
            notalpha.append(i)
    for i in notalpha:
        englishunique.remove(i)
    englishaverage = len(englishunique)

    dutchaverage = 0
    dutchunique = set(list(' '.join(dutch)))
    dutchunique.remove(' ')
    notalpha=[]
    for i in dutchunique:
        if i.isalpha():
            continue
        else:
            notalpha.append(i)
    for i in notalpha:
        dutchunique.remove(i)
    dutchaverage = len(dutchunique)

    threshold = (englishaverage + dutchaverage) / 2

    for i in data:
        i = i[3:]
        dataunique = set(list(' '.join(i)))
        dataunique.remove(' ')
        dataaverage = len(dataunique)


        if dataaverage > threshold:
            feature.append(0)
        else:
            feature.append(1)

    return feature


def bigram(english, dutch, data):
    """
    A feature to calculate the total number of bigrams in the document corpus.

    :param english: English language dataset
    :param dutch: Dutch language dataset.
    :param data: The entire training data
    :return: The feature for each of the samples.
    """
    feature = []
    englishcounter = 0
    for i in english:
        for j in i.split():
            if len(j) == 2:
                englishcounter += 1
    englishcounter = englishcounter / len(english)

    dutchcounter = 0
    for i in dutch:
        for j in i.split():
            if len(j) == 2:
                dutchcounter += 1
    dutchcounter = dutchcounter / len(dutch)

    threshold = (englishcounter + dutchcounter) / 2
    for i in data:
        i = i[3:]
        datacounter = 0
        for j in i.split():
            if len(j) == 2:
                datacounter += 1
        if datacounter > threshold:
            feature.append(0)
        else:
            feature.append(1)
    return feature


def trigram(english, dutch, data):
    """
    A feature to calculate the total number of trigrams in the document corpus.

    :param english: English language dataset
    :param dutch: Dutch language dataset.
    :param data: The entire training data
    :return: The feature for each of the samples.
    """
    feature = []
    englishcounter = 0
    for i in english:
        for j in i.split():
            if len(j) == 3:
                englishcounter += 1
    englishcounter = englishcounter / 15

    dutchcounter = 0
    for i in dutch:
        for j in i.split():
            if len(j) == 3:
                dutchcounter += 1
    dutchcounter = dutchcounter / 15

    threshold = (englishcounter + dutchcounter) / 2

    for i in data:
        i = i[3:]
        datacounter = 0
        for j in i.split():
            if len(j) == 3:
                datacounter += 1
        if datacounter > threshold:
            feature.append(0)
        else:
            feature.append(1)
    return feature


def wordswithrepeatingletters(english, dutch, data):
    """
    A feauture to count the total number of words with repeating letters.

    :param english: English language dataset
    :param dutch: Dutch language dataset.
    :param data: The entire training data
    :return: The feature for each of the samples.
    """
    feature = []
    englishcount = 0
    for i in english:
        for j in i.split():
            englishset = set(j)
            if len(j)-len(englishset) >= 1:
                englishcount += 1
        #englishcount=englishcount/len(i.split())
    englishcount=englishcount/len(english)

    dutchcount = 0
    for i in dutch:
        for j in i.split():
            dutchset = set(j)

            if len(j)-len(dutchset)  >= 1:
                dutchcount += 1
        #dutchcount=dutchcount/len(i.split())
    dutchcount=dutchcount/len(dutch)

    threshold = (englishcount + dutchcount) / 2
    for i in data:
        i = i[3:]
        datacounter = 0
        for j in i.split():

            if len(j)-len(set(j)) >= 1:
                datacounter += 1
        #datacounter=datacounter/len(i.split())

        if datacounter > threshold:
            feature.append(0)
        else:
            feature.append(1)

    return feature
    #return [0,0,0,0,0,0,0,0,0,0]
