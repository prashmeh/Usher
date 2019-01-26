import gensim
import nltk
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class Word2VecLR:
    df = ""
    wv = gensim.models.KeyedVectors.load_word2vec_format(
        "/Users/prashant.mehta/Downloads/GoogleNews-vectors-negative300.bin.gz", binary=True)
    wv.init_sims(replace=True)

    def __init__(self, df):
        self.df = df

    def w2v_tokenize_text(self, text):
        tokens = []
        for sent in nltk.sent_tokenize(text, language='english'):
            for word in nltk.word_tokenize(sent, language='english'):
                if len(word) < 2:
                    continue
                tokens.append(word)
        return tokens

    def word_averaging(self, wv, words):
        all_words, mean = set(), []

        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in wv.vocab:
                mean.append(wv.syn0norm[wv.vocab[word].index])
                all_words.add(wv.vocab[word].index)

        if not mean:
            # logging.warning("cannot compute similarity with no input %s", words)
            # FIXME: remove these examples in pre-processing
            return np.zeros(wv.vector_size, )

        mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

    def word_averaging_list(self, wv, text_list):
        return np.vstack([self.word_averaging(wv, post) for post in text_list])

    def execute(self):
        train, test = train_test_split(self.df, test_size=0.3, random_state=42)

        test_tokenized = test.apply(lambda r: self.w2v_tokenize_text(r['post']), axis=1).values
        train_tokenized = train.apply(lambda r: self.w2v_tokenize_text(r['post']), axis=1).values

        X_train_word_average = self.word_averaging_list(self.wv, train_tokenized)
        X_test_word_average = self.word_averaging_list(self.wv, test_tokenized)

        logreg = LogisticRegression(n_jobs=1, C=1e5)
        logreg = logreg.fit(X_train_word_average, train['tags'])
        y_pred = logreg.predict(X_test_word_average)
        print('accuracy %s' % accuracy_score(y_pred, test.tags))
        print(classification_report(test.tags, y_pred))
