import nltk, os, sys, re, random, string, math
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textModifiers


TRAIN = 'training'
TEST = 'test'

FOOD_RATING = "food"
SERVICE_RATING = "service"
VENUE_RATING = "venue"
OVERALL_RATING = "rating"

FOOD_TEXT = "para1"
SERVICE_TEXT = "para2"
VENUE_TEXT = "para3"
OVERALL_TEXT = "para4"

PARAGRAPH = "paragraph"

FILENAME = "filename"

EmptyPredictions = {}

ConfusionMatrix = {}

def GetFeaturesParagraphRating(reviewSet):
    # Returns a list of (paragraph, rating) tuples based on the following expected order of paragraphs:
    # food, service, venue, rating
    # where paragraph is the dictionary of features
    paragraphRatings = []
    for review in reviewSet:
        if ("para1" in review.keys() and "para2" in review.keys() and
            "para3" in review.keys() and "para4" in review.keys()):
            paragraphRatings.append(({PARAGRAPH: review[FOOD_TEXT]}, GetBinaryRating(review[FOOD_RATING]), review[FILENAME], FOOD_RATING))
            paragraphRatings.append(({PARAGRAPH: review[SERVICE_TEXT]}, GetBinaryRating(review[SERVICE_RATING]), review[FILENAME], SERVICE_RATING))
            paragraphRatings.append(({PARAGRAPH: review[VENUE_TEXT]}, GetBinaryRating(review[VENUE_RATING]), review[FILENAME], VENUE_RATING))
            paragraphRatings.append(({PARAGRAPH: review[OVERALL_TEXT]}, GetBinaryRating(review[OVERALL_RATING]), review[FILENAME], OVERALL_RATING))
    return paragraphRatings


def GetVaderRatings(text):
    # Using the vader sentiment analyzer, returns the dictionary of vader sentiments of the text.
    return SentimentIntensityAnalyzer().polarity_scores(text)


def GetBinaryRating(rating):
    if rating != "":
        return 0 if float(rating) <= 3 else 1
    else:
        return ""


def GetOverallRating(reviewSet, getFileNames=False):
    scores = []
    for review in reviewSet:
        keys = review.keys()
        if "food" in keys and ("rating" in keys or "overall" in keys) \
                and "service" in keys and "venue" in keys:
            food_score = review[FOOD_RATING]
            service_score = review[SERVICE_RATING]
            venue_score = review[VENUE_RATING]
            overall_score = review[OVERALL_RATING]

            if getFileNames:
                features = ({"food_score":food_score, "service_score":service_score,
                        "venue_score":venue_score}, overall_score, review[FILENAME])
            else:
                features = ({"food_score":food_score, "service_score":service_score,
                        "venue_score":venue_score}, overall_score)

            scores.append(features)

    return scores

def GetSpeechTags(allParas):
    text = nltk.word_tokenize(allParas)
    pos = nltk.pos_tag(text)
    speeches = []
    for word, speech in pos:
        speeches.append(speech)
    fd = nltk.FreqDist(speeches)

    return fd

def GetAuthor(reviewSet, getFileNames=False):
    pos_author = []
    for review in reviewSet:
        allParas = textModifiers.GetReviewText(review)
        speechTags = GetSpeechTags(allParas)
        #returns POS used
        if getFileNames:
            features = ({"pos":speechTags}, review["reviewer"], review[FILENAME])
        else:
            features = ({"pos":speechTags}, review["reviewer"])
        pos_author.append(features)

    return pos_author


def TestAndTrainExist(path):
    folders = os.listdir(path)
    return TEST in folders and TRAIN in folders


def BuildDicts(path):
    train = []
    test = []

    if TestAndTrainExist(path):
        #parse the html files
        for file in os.listdir(path + '/' + TEST):
            if not (file.startswith('.')):
                filePath = path + '/' + TEST + '/' + file
                test.append(CleanHtml(filePath))

        for file in os.listdir(path + '/' + TRAIN):
            if not (file.startswith('.')):
                filePath = path + '/' + TRAIN + '/' + file
                train.append(CleanHtml(filePath))
    else:
        #files are in review folders
        allReviews = []
        for i in range(1,4):
            newPath = path + "/Review" + str(i) + "/"
            for folders in os.listdir(newPath):
                if(os.path.isdir(newPath + folders)):
                    for file in os.listdir(newPath + folders):
                        reviewer = folders.split('_')[0]
                        allReviews.append(CleanHtml(newPath + folders + '/' + file, reviewer=reviewer))

        random.shuffle(allReviews)

        #put one review from each reviewer in the test set and the rest in train
        reviewers_in_test = set()
        reviewers_in_train = set()

        for review in allReviews:
            if (review != {} and "reviewer" in review):
                rvwr = review["reviewer"]
                if(rvwr in reviewers_in_train and
                   rvwr in reviewers_in_test):
                    rnd = random.randint(0,1)
                    if(rnd == 1):
                        reviewers_in_train.add(rvwr)
                    else:
                        reviewers_in_test.add(rvwr)

                elif (rvwr in reviewers_in_test):
                    train.append(review)
                    reviewers_in_train.add(rvwr)
                else:
                    test.append(review)
                    reviewers_in_test.add(rvwr)

    return (test, train)


def CleanHtml(htmlPath, reviewer=None):
    fd = open(htmlPath, encoding='utf-8').read()
    # encode('utf-8').strip()

    soup = BeautifulSoup(fd, 'html.parser')
    reviewDict = {}
    reviewDict[FILENAME] = htmlPath.split('/')[-1]

    stop = False
    count = 1
    paras = []
    seen = []
    for paragraphs in soup.findAll(["p", "span"]):
        paragraph = re.sub(r'<[^<]+?>', '', str(paragraphs))
        paragraph = re.sub(r'\n', ' ', str(paragraph)).strip()
        if (paragraph not in seen):
            seen.append(paragraph)
            if paragraph != "\n" and paragraph != "":
                splitParagraph = None
                if not stop:
                    if (':' not in paragraph and paragraph != ""):
                        match = re.match(r"^(\S*)\s([\w\s]+)$", paragraph)
                        if match:
                            splitParagraph = list(match.groups())
                    else:
                        splitParagraph = paragraph.split(':')
                    key = splitParagraph[0].lower().strip()
                    if(key == "written review"):
                        stop = True
                    else:
                        if (key == "overall" or key == OVERALL_RATING):
                            key = OVERALL_RATING
                        if(key == 'reviewer' and reviewer):
                            reviewDict[key] = reviewer
                        else:
                            if(key != "" ):
                                if (len(splitParagraph) > 1):
                                    reviewDict[key] = splitParagraph[1].strip()
                                else:
                                    reviewDict[key] = None
                if stop:
                    if(splitParagraph and len(splitParagraph) > 1):
                        paras.append(splitParagraph[1].strip())
                    else:
                        paras.append(paragraph)
                    count += 1
            paras = [p for p in paras if p != ""]
            for i in range (1, len(paras) + 1):
                reviewDict["para" + str(i)] = paras[i-1]

    return reviewDict


def PredictBinaryRatings(train, test, path, predictMissing=False):
    paraRatingFeaturesTest = GetFeaturesParagraphRating(test)

    tp = tn = fp = fn = 0

    num_correct = 0
    num_total = 0
    for feature, label, filename, topic in paraRatingFeaturesTest:
        vader_ratings = GetVaderRatings(feature["paragraph"])
        predict = 0 if vader_ratings["neg"] > vader_ratings["pos"] else 1

        if predictMissing and TestAndTrainExist(path) and label == "" and topic != OVERALL_RATING:
            print("   {}: {} rating = {}".format(filename, topic, predict))

        if predict == label:
            if(predict == 1):
                tp += 1
            else:
                tn += 1
            num_correct += 1
        else:
            if(predict == 0):
                fn += 1
            else:
                fp += 1
        num_total += 1
    return (num_correct/num_total, tp, tn, fp, fn)


def PredictOverallRatings(train, test, path, predictMissing=False):
    # Given the train set and test set, return the AveRMS score for predicting overall ratings of reviews
    getOverallRatingTrain = GetOverallRating(train)
    getOverallRatingTest = GetOverallRating(test, getFileNames=True)

    NBClassifier = nltk.NaiveBayesClassifier.train(getOverallRatingTrain)
    correct = 0
    count = 0
    predict_actuals = []
    for feature, label, filename in getOverallRatingTest:
        correctLabel = label
        classifiedLabel = NBClassifier.classify(feature)
        if predictMissing and TestAndTrainExist(path) and label == "":
            print("   {}: overall rating = {}".format(filename, classifiedLabel))
        if (classifiedLabel == correctLabel):
            correct += 1
        count += 1
        if correctLabel != "":
            predict_actuals.append((float(classifiedLabel), float(correctLabel)))
    #print("   Predict overall rating accuracy: {}".format(correct / count))
    return (RMS(predict_actuals), correct / count)


def CompareAuthorTest(authorDict, posCounts):
    toReturn = ()
    min = sys.maxsize
    for label, counts in authorDict.items():
        if counts != []:
            diffCounter = 0
            for pos, count in posCounts:
                for testPos, testCount in counts:
                    if(testPos == pos):
                        diffCounter += abs(count - testCount)

            if diffCounter < min:
                min = diffCounter
                toReturn = label

    return toReturn


def GetFeaturesDistinctWordsSentiment(distinct_bad, distinct_good):
    features = []
    for word, freq in distinct_bad:
        features.append(({word: freq}, 0))
    for word,freq in distinct_good:
        features.append(({word: freq}, 1))
    return features


def PredictBinaryFromWordFreqs(distinct_0, distinct_1, test):
    train_set = GetFeaturesDistinctWordsSentiment(distinct_0, distinct_1)
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    num_correct = 0
    num_total = 0
    for review in test:
        if (review != {} and OVERALL_RATING in review):
            non_stopwords = nltk.FreqDist(textModifiers.RemoveStopwords(textModifiers.GetReviewText(review)))
            classified = classifier.classify(non_stopwords)
            if classified == GetBinaryRating(review[OVERALL_RATING]):
                num_correct += 1
            num_total += 1

    return num_correct / num_total


def PredictAuthor(train, test, path, predictMissing=False):
    # Given the train set and test set, return the AveRMS score for predicting the author of reviews
    getAuthorFeaturesTrain = GetAuthor(train)
    getAuthorFeaturesTest = GetAuthor(test, getFileNames=True)
    mostCommonNum = 30  #top 30 most common pos tags

    authorDict = {}
    seen = []
    for feature, label in getAuthorFeaturesTrain:
        mostCommon = feature["pos"].most_common(mostCommonNum)

        if(label in seen):
            after = []
            for items, count in authorDict[label]:
                for mcItems, mcCount in mostCommon:
                    if(items == mcItems):
                        newCount = math.floor((count + mcCount) / 2)
                        newTuple = (mcItems, newCount)
                        after.append(newTuple)
            authorDict[label] = after
        else:
            seen.append(label)
            authorDict[label] = mostCommon

    count = 0
    correct = 0
    for feature, label, filename in getAuthorFeaturesTest:
        predictedAuthor = CompareAuthorTest(authorDict,feature["pos"].most_common(mostCommonNum))
        if predictMissing and TestAndTrainExist(path) and label == "":
            print("   {}: reviewer = {}".format(filename, predictedAuthor))
        BuildConfMatrix(label, predictedAuthor)
        if(predictedAuthor == label):
            correct += 1
        count += 1
    return correct / count

def BuildConfMatrix(actual, predicted):
    if actual not in ConfusionMatrix:
        ConfusionMatrix[actual] = {}
    if predicted not in ConfusionMatrix[actual]:
        ConfusionMatrix[actual][predicted] = 1
    else:
        ConfusionMatrix[actual][predicted] += 1
    return


def RMS(prediction_actuals):
    # Returns the average root-mean-square of the given values
    # predition_actuals is a list of (prediction, actual) tuples
    return math.sqrt(sum([pow(p - a, 2) for p, a in prediction_actuals]) / len(prediction_actuals))


def main():
    path = ""
    if(len(sys.argv) > 1):
        if("-h" in sys.argv):
            print("Usage: restaurants.py DATA_DIR")
            print("The people who worked on this project are: Logan Williams and Justin Postigo")
            if("-h" == sys.argv[1] and len(sys.argv) > 2):
                path = sys.argv[2]
            elif("-h" == sys.argv[2] and len(sys.argv) > 2):
                path = sys.argv[1]
        else:
            path = sys.argv[1]
    else:
        print("Usage: restaurants.py DATA_DIR")
        sys.exit(1)

    if TestAndTrainExist(path):
        print("Predictions for missing values:")

    # Run tests 5 times with new test/train sets each iteration and compute averages
    binaryRatingsAccuracy = 0
    binaryWordFreqsAccuracy = 0
    mostCommonWordFreqs = 10  # Change to use more/less words for exercise 2
    distinct_0, distinct_1 = None, None
    overallRatingsAccuracy = 0
    authorAccuracy = 0
    num_trials = 5
    overallRatingRMS = 0
    precision = []
    recall = []
    f1_score = []
    for i in range(num_trials):
        test,train = BuildDicts(path)
        brAccuracy, tp, tn, fp, fn = PredictBinaryRatings(train, test, path, predictMissing=i==0)
        binaryRatingsAccuracy += brAccuracy
        precision.append(tp/(tp + fp))
        recall.append(tp/(tp+fn))

        f1_score.append((2*precision[i]*recall[i]) / (precision[i] + recall[i]))

        distinct_0, distinct_1 = textModifiers.GetSentimentWords(train, mostCommonWordFreqs)
        binaryWordFreqsAccuracy += PredictBinaryFromWordFreqs(distinct_0, distinct_1, test)

        averms, ORAccuracy = PredictOverallRatings(train, test, path, predictMissing=i==0)
        overallRatingRMS += averms
        overallRatingsAccuracy += ORAccuracy
        authorAccuracy += PredictAuthor(train, test, path, predictMissing=i==0)


    binaryRatingsAccuracy /= num_trials
    binaryWordFreqsAccuracy /= num_trials
    overallRatingsAccuracy /= num_trials
    authorAccuracy /= num_trials
    overallRatingRMS /= num_trials

    precisionAverage = sum(precision)/num_trials
    recallAverage = sum(recall)/num_trials
    f1_scoreAverage = sum(f1_score)/num_trials

    if TestAndTrainExist(path):
        print()
    # Exercise 1 -- Predict the binary rating of each paragraph regardless of subject, assume correct order for ratings.
    #print("Average RMS error of 5 trials for predicting binary ratings of individual paragraphs: {}"
    #      .format(binaryRatingsAccuracy))
    print("Exercise 1:")
    print("   Average predict binary rating accuracy: {}".format(binaryRatingsAccuracy))
    print("   Average predict binary rating precision: {}".format(precisionAverage))
    print("   Average predict binary rating recall: {}".format(recallAverage))
    print("   Average predict binary rating F1 Score: {}".format(f1_scoreAverage))
    print()

    # Exercise 2 -- Use NLTK functions and corpora to discover three interesting phenomena about the restaurant corpus.
    # Use machine learning to prove this. Discuss your results.
    #  - Remove stopwords, count freqdist on rest of words, split reviews according to binary rating and compare
    #    the non-distinct words, then use these most common compared word freqs as features in a NB classifier
    print("Exercise 2:")
    print("   Most common word count differences for good ratings: {}".format(distinct_1))
    print("   Most common word count differences for bad ratings: {}".format(distinct_0))
    print("   Accuracy predicting binary rating using most common word count differences: {}"
          .format(binaryWordFreqsAccuracy))
    print()

    # Exercise 3 -- Predict the overall rating of each review (1-5) considering all information from the review, except
    # for the final rating number.
    print("Exercise 3: ")
    print("   Average predict overall rating accuracy: {}".format(overallRatingsAccuracy))
    print("   Average RMS error for overall rating: {}".format(overallRatingRMS))
    print()
    # Exercise 4 -- Predict the author of each review.
    #print("Average RMS error of 5 trials for predicting the author of each review: {}"
    #      .format(authorAccuracy))
    print("Exercise 4: ")
    print("   Average predict authorship accuracy: {}".format(authorAccuracy))


if __name__ == "__main__":
    main()

