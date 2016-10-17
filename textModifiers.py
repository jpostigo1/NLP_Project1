from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize, FreqDist
from nltk.stem.snowball import SnowballStemmer

def RemoveStopwords(text):
    # Returns a list of words in text excluding stopwords
    # where text is a single string
    stop = set(stopwords.words("english"))
    return [word for word in word_tokenize(text) if word not in stop]

def GetMostCommonStems(words):
    # Stem a list of words and return the sorted list of word frequencies, not including stop words
    # where words is a list of strings
    stemmer = SnowballStemmer("english")
    freqs = FreqDist([stemmer.stem(word.lower()) for word in words])
    return dict(freqs)

def GetDistinctWords(words1, words2):
    # Returns a tuple of the distinct words from each dictionary
    # words1 and words2 are dictionaries of {"word": freq} pairs
    distinct_words1 = [(word, freq) for word, freq in words1.items() if word not in words2.keys()]
    distinct_words2 = [(word, freq) for word, freq in words2.items() if word not in words1.keys()]
    return (distinct_words1, distinct_words2)


def func(set, n=5):
    # Given a set of review dictionaries (such as test or train),
    # return the `n` most frequent words distinct to each overall rating
    pass


s1 = word_tokenize("One two three four one two three four one two three four")
s2 = word_tokenize("One three five six one four six five five five")

print(GetDistinctWords(GetMostCommonStems(s1), GetMostCommonStems(s2)))
