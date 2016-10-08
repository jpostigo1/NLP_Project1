#Language classification demo using NLTK and python
#uses the Universal Declaration of Human Rights (udhr) corpus, and Naive Bayes classifier from NLTK
#by Foaad Khosmood / Cal Poly / Oct 5, 2012
#updated Python3 print() statements Sep / 2014
#updated for Python3 Oct / 2014

#importing standard packages + NLTK
import random,operator,nltk
from nltk.corpus import udhr

#ratio of train:test documents
trainRatio = 3


#minimum word length we want to consider
ml = 3

#languages we are interested in (over 300 available in the udhr corpus) MUST be "Latin1" coding
languages = ['Spanish_Espanol', 'Welsh_Cymraeg', 'Afrikaans','Basque_Euskara','Danish_Dansk','Dutch_Nederlands','Finnish_Suomi','French_Francais','German_Deutsch','English','Italian']


wordsToUseAsFeatures = []

#Define a function that produces features from a given object, in this case one word
#Three string features are extracted per word: first two letters, last letter and last three letters
#Note that featuresets are dictionaries. That's what the classifier takes as input
def langFeatures(word):
	features = {'first_two':word[:2], 'last_letter':word[-1],'last_three':word[-3:]}
	
	if word in wordsToUseAsFeatures:
		features['word-'+word]=True
	
	return features

# a function operating on training words only, that could help us get more features
# in this case, we are finding the most frequent whole words in the trainig set
def getMoreFeatures(trainWords):
	moreFeatures = []
	for l in languages:
		langWords = [w for (w,l) in trainWords]
		fdist = nltk.FreqDist(langWords)
		for w in list(fdist.keys()): #fdist.N() // 5]:
			moreFeatures.append(w)
	return moreFeatures
	

#use Python's functional features to get a big list of (word,Langauge) from languages, we are interested in
#words = reduce(operator.add, map(lambda L: ([(w.lower(),L) for w in udhr.words(L+'-Latin1') if len(w) >= ml]),languages),[])

words = []
allLists = [[(w.lower(),L) for w in udhr.words(L+'-Latin1') if len(w) >= ml] for L in languages]
for L in allLists:
	words.extend(L)


#engWords, afrWords, itaWords = udhr.words('English-Latin1'), udhr.words('Afrikaans-Latin1'), udhr.words('Italian-Latin1')
#words = [(w,'English') for w in engWords] + [(w,'Afrikaans') for w in afrWords] + [(w,'Italian') for w in itaWords]
#words = [(w,l) for (w,l) in words if len(w) >= ml]

#(word, Langauge) tuples are still in file access order. This randomizes them
random.shuffle(words)

#split into training and test words still just (w,l) tuples.
splitPoint = len(words)//trainRatio
testWords, trainWords = words[:splitPoint],words[splitPoint:]

#Analysis on training set (you are not allowed to learn anything from the test set)
wordsToUseAsFeatures.extend(getMoreFeatures(trainWords))
		
#convert the (word,L) -> (features(word),L) for both training and test sets
test = [(langFeatures(w),l) for (w,l) in testWords]
train = [(langFeatures(w),l) for (w,l) in trainWords]

#NLTK's built-in implementation of the Naive Bayes classifier is trained
classifier = nltk.NaiveBayesClassifier.train(train)

#Other classifiers easily available from NLTK: Decision Trees and MaxEnt
#classifier = nltk.MaxentClassifier.train(train,max_iter=5)
#classifier = nltk.DecisionTreeClassifier.train(train,entropy_cutoff=0.1)

#now, it is tested on the test set and the accuracy reported
print("Accuracy: ",nltk.classify.accuracy(classifier,test))

#this is a nice function that reports the top most impactful features the NB classifier found
#It works for Maxent too, but it is not defined for DecisionTrees. So comment it out for DTs.
print(classifier.show_most_informative_features(20)) 
