from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize


s1 = "The restaurant wasn't very good. I thought the food was okay, but overall, I was not impressed with the experience.\
The restaurant serves Italian food. They have seven employees who all are trained in the food industry."


sid = SentimentIntensityAnalyzer()
for sentence in sent_tokenize(s1):
     print(sentence)
     ss = sid.polarity_scores(sentence)
     for k in sorted(ss):
         print('{0}: {1}, '.format(k, ss[k]), end='')
     print()