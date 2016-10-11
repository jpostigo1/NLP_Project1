import nltk, os, sys, re
from bs4 import BeautifulSoup

def CleanHtml(path):
    #goes through the folders in path
    for root, folders, files in os.walk(path):
        for folder in folders:
            #print(folder)
            #get each 'onlinetext.html' file in eat folder
            #not sure if this is the most elegant way but its working
            for file in os.listdir(root + folder):
                print(file)
                #again, a very non-elegant way to read the file
                fd = open(root + folder + "/" + file).read()
                soup = BeautifulSoup(fd, 'html.parser')

                #print(soup.prettify())
                #works if each line is separated by a new line
                for paragraphs in soup.findAll("p"):
                    for line in paragraphs:
                        #removes any additional html tags
                        toRemove = re.compile('<[^<]+?>')
                        line = re.sub(toRemove, '', str(line))
                        print(line)

                #adding breaks to only test one file
                break
            break
        break


    return

def main():
    if(len(sys.argv) >= 2):
        if("-h" in sys.argv):
            print("Usage: restaurants.py DATA_DIR")
            print("The people who worked on this project are: Logan Williams and Justin Postigo")
    else:
        print("Usage: restaurants.py DATA_DIR")
    #just a test path
    path = "./Restaurant_Project_2016/Review1/"
    CleanHtml(path)

    return

if __name__ == "__main__":
    main()