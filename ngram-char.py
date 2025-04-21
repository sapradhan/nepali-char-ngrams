from sklearn.feature_extraction.text import CountVectorizer
import re

corpus = ["nep-np_web_2016_100K/nep-np_web_2016_100K-sentences.txt"]

vectorizer = CountVectorizer(input="filename", analyzer='char_wb', ngram_range=(1, 3), 
                             preprocessor=lambda x: re.sub(r'[^\u0900-\u097F]', '', x))
X = vectorizer.fit_transform(corpus)
ngrams = vectorizer.get_feature_names_out()
counts = X.toarray().sum(axis=0)

with open("nep-np_web_2016_100K/nep-np_web_2016_100K-ngram.tsv", "w", encoding="utf-8") as f:
    for e in zip(ngrams, counts):
        # limit to ngrams of interest - all unigrams
        # bigrams that end with halant, 
        # trigrams are conjunts 
        if len(e[0]) == 1 or e[0][1] == '्': 
            f.write('\t'.join([e[0], str(int(e[1]))]) + '\n')
