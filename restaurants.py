import nltk, os, sys, re, random, string, math
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy

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


def GetFeaturesParagraphRating(reviewSet):
    # Returns a list of (paragraph, rating) tuples based on the following expected order of paragraphs:
    # food, service, venue, rating
    # where paragraph is the dictionary of features
    paragraphRatings = []
    for review in reviewSet:
        if ("para1" in review.keys() and "para2" in review.keys() and
            "para3" in review.keys() and "para4" in review.keys()):
            paragraphRatings.append(({PARAGRAPH: review[FOOD_TEXT]}, GetBinaryRating(review[FOOD_RATING])))
            paragraphRatings.append(({PARAGRAPH: review[SERVICE_TEXT]}, GetBinaryRating(review[SERVICE_RATING])))
            paragraphRatings.append(({PARAGRAPH: review[VENUE_TEXT]}, GetBinaryRating(review[VENUE_RATING])))
            paragraphRatings.append(({PARAGRAPH: review[OVERALL_TEXT]}, GetBinaryRating(review[OVERALL_RATING])))
    return paragraphRatings


def GetVaderRatings(text):
    # Using the vader sentiment analyzer, returns the dictionary of vader sentiments of the text.
    return SentimentIntensityAnalyzer().polarity_scores(text)


def GetBinaryRating(rating):
    return 0 if float(rating) <= 3 else 1


def GetOverallRating(reviewSet):
    scores = []
    for review in reviewSet:
        keys = review.keys()
        if "food" in keys and ("rating" in keys or "overall" in keys) \
                and "service" in keys and "venue" in keys:
            food_score = review[FOOD_RATING]
            service_score = review[SERVICE_RATING]
            venue_score = review[VENUE_RATING]
            overall_score = review[OVERALL_RATING]

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

def GetAuthor(reviewSet):
    pos_author = []
    for review in reviewSet:
        allParas = ""
        for key in review.keys():
            #get all paragraphs regardless of how many
            if "para" in key:
                allParas += " " + review[key]

        speechTags = GetSpeechTags(allParas)
        #returns POS used
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
            filePath = path + '/' + TEST + '/' + file
            test.append(CleanHtml(filePath))

        for file in os.listdir(path + '/' + TRAIN):
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
        for review in allReviews:
            if (review != {}):
                if (review["reviewer"] in reviewers_in_test):
                    train.append(review)
                else:
                    test.append(review)
                    reviewers_in_test.add(review["reviewer"])

    return (test, train)


def CleanHtml(htmlPath, reviewer=None):
    fd = open(htmlPath, encoding='utf-8').read()
    soup = BeautifulSoup(fd, 'html.parser')
    reviewDict = {}
    #order of paragraphs:
    #reviewer, name, address, city, food, service, venue,
    #rating, written review, 4 paragraphs
    stop = False
    count = 1
    paras = []
    seen = []
    for paragraphs in soup.findAll(["p", "span"]):
        #print("paragraphs: |{}|".format(paragraphs))
        paragraph = re.sub(r'<[^<]+?>', '', str(paragraphs))
        paragraph = re.sub(r'\n', ' ', str(paragraph)).strip()
        if (paragraph not in seen):
            seen.append(paragraph)
            if paragraph != "\n" and paragraph != "":
                #print("paragraph: |{}|".format(paragraph))
                #paragraph = paragraph.encode('ascii', 'ignore').decode('utf-8', 'ignore')
                splitParagraph = None
                if not stop:
                    if (':' not in paragraph and paragraph != ""):
                        match = re.match(r"^(\S*)\s([\w\s]+)$", paragraph)
                        if match:
                            splitParagraph = list(match.groups())
                    else:
                        splitParagraph = paragraph.split(':')
                    #print("splitParagraph: |{}|".format(splitParagraph))
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


def PredictBinaryRatings(train, test):
    paraRatingFeaturesTrain = GetFeaturesParagraphRating(train)
    paraRatingFeaturesTest = GetFeaturesParagraphRating(test)

    #NBClassifier = nltk.NaiveBayesClassifier.train(paraRatingFeaturesTrain)
    #MEClassifier = nltk.MaxentClassifier.train(paraRatingFeaturesTrain, max_iter=5)
    #DTClassifier = nltk.DecisionTreeClassifier.train(paraRatingFeaturesTrain, entropy_cutoff=0.1)

    # for feature, label in paraRatingFeaturesTest:
    #    print("Features: {}\nClassified as: {}\nCorrect label: {}\n\n".format(feature, NBClassifier.classify(feature), label))

    num_correct = 0
    num_total = 0
    predict_actuals = [] # list of (predicted_label, acutal_label) tuples for RMS calculation
    for feature, label in paraRatingFeaturesTest:
        vader_ratings = GetVaderRatings(feature["paragraph"])
        predict = 0 if vader_ratings["neg"] > vader_ratings["pos"] else 1
        if predict == label:
            num_correct += 1
        num_total += 1
        predict_actuals.append((predict, label))
    #print("Accuracy for Vader: {}".format(float(num_correct) / num_total))
    #print("Average RMS error for Vader: {}".format(AveRMS(predict_actuals)))
    return AveRMS(predict_actuals)


def PredictOverallRatings(train, test):
    # Given the train set and test set, return the AveRMS score for predicting overall ratings of reviews
    getOverallRatingTrain = GetOverallRating(train)
    getOverallRatingTest = GetOverallRating(test)

    NBClassifier = nltk.NaiveBayesClassifier.train(getOverallRatingTrain)
    correct = 0
    count = 0
    predict_actuals = []
    for feature, label in getOverallRatingTest:
        correctLabel = label
        classifiedLabel = NBClassifier.classify(feature)
        #print("Features: {}\nClassified as: {}\nCorrect label: {}\n\n".format(feature, NBClassifier.classify(feature), label))
        if(classifiedLabel == correctLabel):
            correct += 1
        count += 1
        #print(classifiedLabel, correctLabel)
        predict_actuals.append((float(classifiedLabel), float(correctLabel)))
    #print(correct / count)

    return AveRMS(predict_actuals)




def CompareAuthorTest(authorDict, posCounts):
    #print("poscounts: ", posCounts)
    toReturn = ()
    min = sys.maxsize
    for label, counts in authorDict.items():
        if counts != []:
            diffCounter = 0
            #print("counts: ", counts)
            for pos, count in posCounts:
                for testPos, testCount in counts:
                    if(testPos == pos):
                        #print(pos, count, testCount)
                        diffCounter += abs(count - testCount)

            if diffCounter < min:
                min = diffCounter
                toReturn = label

    return toReturn

def PredictAuthor(train, test):
    # Given the train set and test set, return the AveRMS score for predicting the author of reviews
    getAuthorFeaturesTrain = GetAuthor(train)
    getAuthorFeaturesTest = GetAuthor(test)

    authorDict = {}
    seen = []
    for feature, label in getAuthorFeaturesTrain:
        #top 20 most common pos tags
        mostCommon = feature["pos"].most_common(20)
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
    predict_actuals = []
    for feature, label in getAuthorFeaturesTest:

        predictedAuthor = CompareAuthorTest(
            authorDict,feature["pos"].most_common(20))
        if(predictedAuthor == label):
            predict_actuals.append((1,1))
            correct += 1
        else:
            predict_actuals.append((0,1))
        count += 1
    #result = correct / count
    #print(predict_actuals)
    return AveRMS(predict_actuals)


def AveRMS(prediction_actuals):
    # Returns the average root-mean-square of the given values
    # predition_actuals is a list of (prediction, actual) tuples
    return math.sqrt(sum([pow(p - a, 2) for p, a in prediction_actuals]) / len(prediction_actuals))


def AverageFiveTrials(func):
    num_trials = 5
    results = []
    for i in range(5):
        results.append(func())
    return sum(results) / num_trials


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

    test,train = BuildDicts(path)
    #print("Test: {}\n\nTrain: {}".format(test, train))


    # Exercise 1 -- Predict the binary rating of each paragraph regardless of subject, assume correct order for ratings.
    #print("Average RMS error of 5 trials for predicting binary ratings of individual paragraphs: {}"
    #      .format(1 - AverageFiveTrials(lambda: PredictBinaryRatings(train, test))))

    # Exercise 2 -- Use NLTK functions and corpora to discover three interesting phenomena about the restaurant corpus.
    # Use machine learning to prove this. Discuss your results.
    #
    #  - Could possibly use sentimentAnalysis.py to find interesting stats on subjectivity/objectivity of reviews
    #  - Could remove stopwords, count freqdist on rest of words, split into good/bad or subj/obj reviews and remove
    #    the non-distinct words, use as features in a classifier
    #  - ...?

    # Exercise 3 -- Predict the overall rating of each review (1-5) considering all information from the review, except
    # for the final rating number.
    #PredictOverallRatings(train, test)
    print("Average RMS error of 5 trials for predicting overall rating of each review: {}"
          .format(1 - AverageFiveTrials(lambda: PredictOverallRatings(train, test))))

    # Exercise 4 -- Predict the author of each review.
    #PredictAuthor(train, test)
    #print("Average RMS error of 5 trials for predicting the author of each review: {}"
    #      .format(1 - AverageFiveTrials(lambda: PredictAuthor(train, test))))


if __name__ == "__main__":
    main()
