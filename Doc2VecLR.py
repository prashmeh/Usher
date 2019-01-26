import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from gensim.models import doc2vec, Doc2Vec
from sklearn import utils

tqdm.pandas(desc="progress-bar")


class Doc2VecLR:
    df = ""

    def __init__(self, df):
        self.df = df

    def execute(self):
        X_train, X_test, y_train, y_test = train_test_split(self.df['post'], self.df['tags'], random_state=0, test_size=0.3)
        X_train = self.label_sentences(X_train, 'Train')
        X_test = self.label_sentences(X_test, 'Test')
        all_data = X_train + X_test
        model = self.train_all_data(all_data)
        train_vectors_dbow = self.get_vectors(model, len(X_train), 300, 'Train')
        test_vectors_dbow = self.get_vectors(model, len(X_test), 300, 'Test')
        logreg = LogisticRegression(n_jobs=1, C=1e5)
        logreg.fit(train_vectors_dbow, y_train)
        logreg = logreg.fit(train_vectors_dbow, y_train)
        y_pred = logreg.predict(test_vectors_dbow)
        print('accuracy %s' % accuracy_score(y_pred, y_test))
        print(classification_report(y_test, y_pred))

    def train_all_data(self, all_data):
        model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
        model_dbow.build_vocab([x for x in tqdm(all_data)])

        for epoch in range(30):
            model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
            model_dbow.alpha -= 0.002
            model_dbow.min_alpha = model_dbow.alpha
        return model_dbow

    def label_sentences(self, corpus, label_type):
        """
        Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
        We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
        a dummy index of the post.
        """
        labeled = []
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            labeled.append(doc2vec.TaggedDocument(v.split(), [label]))
        return labeled

    def get_vectors(self, model, corpus_size, vectors_size, vectors_type):
        """
        Get vectors from trained doc2vec model
        :param doc2vec_model: Trained Doc2Vec model
        :param corpus_size: Size of the data
        :param vectors_size: Size of the embedding vectors
        :param vectors_type: Training or Testing vectors
        :return: list of vectors
        """
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = model.docvecs[prefix]
        return vectors


