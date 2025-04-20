from sklearn.feature_extraction.text import CountVectorizer

corpus = ["nep-np_web_2016_100K/nep-np_web_2016_100K-sentences.txt"]

vectorizer = CountVectorizer(input="filename", analyzer='char_wb', ngram_range=(1, 3))
X = vectorizer.fit_transform(corpus)
ngrams = vectorizer.get_feature_names_out()
counts = X.toarray().sum(axis=0)

with open("nep-np_web_2016_100K/nep-np_web_2016_100K-ngram.tsv", "w", encoding="utf-8") as f:
    for e in zip(ngrams, counts):
        f.write('\t'.join([e[0], str(int(e[1]))]) + '\n')
