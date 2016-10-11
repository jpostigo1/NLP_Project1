import nltk, os, sys, re
from bs4 import BeautifulSoup

TRAIN = 'training'
TEST = 'test'

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

        return
    return (test, train)

def GetPath(path):
    folders = os.listdir(path)
    return TEST in folders and TRAIN in folders

def CleanHtml(htmlPath):
    fd = open(htmlPath).read()
    soup = BeautifulSoup(fd, 'html.parser')
    reviewDict = {}
    #order of paragraphs:
    #reviewer, name, address, city, food, service, venue,
    #rating, written review, 4 paragraphs
    stop = False
    count = 1
    for paragraphs in soup.findAll("p"):
        paragraph = re.sub(r'<[^<]+?>', '', str(paragraphs))
        if not stop:
            splitParagraph = paragraph.split(':')
            if(splitParagraph[0].lower() == "written review"):
                stop = True
            else:
                reviewDict[splitParagraph[0].lower()] = splitParagraph[1].strip()
        elif stop:
            reviewDict["para" + str(count)] = paragraph
            count += 1

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

    BuildDicts(path)



    return

if __name__ == "__main__":
    main()