import nltk, os, sys, re, random, string
from bs4 import BeautifulSoup

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
        paragraphRatings.append(({PARAGRAPH: review[FOOD_TEXT]}, review[FOOD_RATING]))
        paragraphRatings.append(({PARAGRAPH: review[SERVICE_TEXT]}, review[SERVICE_RATING]))
        paragraphRatings.append(({PARAGRAPH: review[VENUE_TEXT]}, review[VENUE_RATING]))
        paragraphRatings.append(({PARAGRAPH: review[OVERALL_TEXT]}, review[OVERALL_RATING]))
    return paragraphRatings


def GetBinaryRating(rating):
    if rating <= 3:
        return 0
    else:
        return 1


def GetOverallRating(reviewSet):
    #returns a predicted rating for 'overall' or 'rating'
    #tuples like (food_rating, score), etc.
    scores = []
    for review in reviewSet:
        food_score = review[FOOD_RATING]
        service_score = review[SERVICE_RATING]
        venue_score = review[VENUE_RATING]

        foodTuple = (FOOD_RATING, food_score)
        serviceTuple = (SERVICE_RATING, service_score)
        venueTuple = (VENUE_RATING, venue_score)

        overall_score = review[OVERALL_RATING]

        scores.append(((foodTuple, serviceTuple, venueTuple), overall_score))


    return scores

def GetAuthor(reviewSet):
    #tuples of (paras[0:4], reviewer)
    #order: food,service,venue,overall
    paras_author = []
    for review in reviewSet:
        allParas = review[FOOD_TEXT] + "\n" + review[SERVICE_TEXT] + "\n" + review[VENUE_TEXT] + "\n" + review[OVERALL_TEXT]

        paras_author.append((allParas, review["reviewer"]))
    return paras_author


def BuildDicts(path):
    train = []
    test = []

    testAndTrain = GetPath(path)

    if testAndTrain:
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


def GetPath(path):
    folders = os.listdir(path)
    return TEST in folders and TRAIN in folders


def CleanHtml(htmlPath, reviewer=None):
    print(htmlPath)
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
    print("Test: {}\n\nTrain: {}".format(test, train))


    return

if __name__ == "__main__":
    main()