import nltk, os, sys, re, random, string
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
        ratio = 4
        for i in range(1,4):
            newPath = path + "/Review" + str(i) + "/"
            for folders in os.listdir(newPath):
                if(os.path.isdir(newPath + folders)):
                    for file in os.listdir(newPath + folders):
                        reviewer = folders.split('_')[0]
                        CleanHtml(newPath + folders + '/' + file, reviewer=reviewer)

        splitPoint = len(allReviews)//ratio
        random.shuffle(allReviews)
        test, train = allReviews[:splitPoint], allReviews[splitPoint:]
    return (test, train)

def GetPath(path):
    folders = os.listdir(path)
    return TEST in folders and TRAIN in folders

def CleanHtml(htmlPath, reviewer=None):
    fd = open(htmlPath).read()
    soup = BeautifulSoup(fd, 'html.parser')
    reviewDict = {}
    #order of paragraphs:
    #reviewer, name, address, city, food, service, venue,
    #rating, written review, 4 paragraphs
    stop = False
    count = 1
    paras = []
    for paragraphs in soup.findAll("p"):
        paragraph = re.sub(r'<[^<]+?>', '', str(paragraphs))
        #paragraph = paragraph.encode('ascii', 'ignore').decode('utf-8', 'ignore')
        splitParagraph = ""
        if not stop:
            splitParagraph = paragraph.split(':')
            key = splitParagraph[0].lower()
            if(key == "written review"):
                stop = True
            else:
                if(key == 'reviewer' and reviewer):
                    reviewDict[key] = reviewer
                else:
                    if(key != ""):
                        reviewDict[key] = splitParagraph[1].strip()
        if stop:
            if(len(splitParagraph) > 1):
                paras.append(splitParagraph[1])
            else:
                paras.append(paragraph)
            count += 1
    paras = [p for p in paras if p != ""]
    for i in range (1, len(paras) - 1):
        reviewDict["para" + str(i)] = paras[i-1]
    print(sorted(reviewDict.items()))
    return sorted(reviewDict.items())
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