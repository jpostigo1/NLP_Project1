from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize, FreqDist, pos_tag
from nltk.stem.snowball import SnowballStemmer

def RemoveStopwords(text):
    # Returns a list of words in text excluding stopwords
    # where text is a single string
    stop = set(stopwords.words("english"))
    return [word for word in word_tokenize(text) if word not in stop]


def GetComparativeFreqs(words1, words2):
    # words1 and words2 are lists of dictionaries of {word: freq} entries
    # return two lists of dictionaries with the adjusted frequencies
    # for example, if words1 is [{"dog": 5}] and words2 is [{"dog": 2}]
    # then return words1 as {("dog": 3}] and words2 as []
    new_words1 = {}
    for word, freq in words1.items():
        if word in words2.keys():
            new_freq = freq - words2[word]
            if new_freq > 0:
                new_words1[word] = new_freq
        else:
            new_words1[word] = freq

    new_words2 = {}
    for word, freq in words2.items():
        if word in words1.keys():
            new_freq = freq - words1[word]
            if new_freq > 0:
                new_words2[word] = new_freq
        else:
            new_words2[word] = freq

    return new_words1, new_words2


def GetMostCommonWordFreqs(words, n=10):
    return sorted(words.items(), key=lambda t: -t[1])[:n]


def SplitOnOverallRating(set):
    # Return two sets of dictionaries split from `set` based on overall ratings
    set_0 = []
    set_1 = []
    for review in set:
        if "rating" in review and float(review["rating"]) <= 3:
            set_0.append(review)
        else:
            set_1.append(review)
    return set_0, set_1


def GetReviewText(review):
    # Returns just the paragraphs from a given review as a single string.
    allParas = ""
    for key in review.keys():
        # get all paragraphs regardless of how many
        if "para" in key:
            allParas += "\n" + review[key]
    return allParas


def GetSentimentWords(set, n=5):
    # Given a set of review dictionaries (such as test or train),
    # return the `n` most frequent words compared to good(4,5)/bad(1,2,3) ratings
    set_0, set_1 = SplitOnOverallRating(set)

    words_0 = [word.lower() for review in set_0 for word in RemoveStopwords(GetReviewText(review))]
    freqDict0 = dict(FreqDist([word.lower() for word in words_0]))

    words_1 = [word.lower() for review in set_1 for word in RemoveStopwords(GetReviewText(review))]
    freqDict1 = dict(FreqDist([word.lower() for word in words_1]))

    comparedFreqs0, comparedFreqs1 = GetComparativeFreqs(freqDict0, freqDict1)

    most_common0 = GetMostCommonWordFreqs(comparedFreqs0, n)
    most_common1 = GetMostCommonWordFreqs(comparedFreqs1, n)

    return most_common0, most_common1
