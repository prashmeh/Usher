# from nltk.corpus import stopwords
import re

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# %matplotlib inline
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from BOWKeras import BOWKeras
from Doc2VecLR import Doc2VecLR
from Word2VecLR import Word2VecLR

nltk.download('stopwords')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
# df = pd.read_csv('/Users/prashant.mehta/Downloads/stack-overflow-data.csv')
df = pd.read_csv('zendeskData.csv')
df = df[pd.notnull(df['tags'])]


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = BeautifulSoup(text, 'lxml').text  # HTML decoding
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
    return text


df['post'] = df['post'].apply(clean_text)
# Training the data
X = df.post
y = df.tags
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Naive Bayes Classifier for Multinomial Models
def naiveBayesClassifier(df, X_train, X_test, y_train, y_test):
    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                   ])
    nb.fit(X_train, y_train)

    # %%time
    y_pred = nb.predict(X_test)
    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))


def SVM(df, X_train, X_test, y_train, y_test):
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])
    sgd.fit(X_train, y_train)

    # % % time

    y_pred = sgd.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))


def logisticRegression(df, X_train, X_test, y_train, y_test):
    logreg = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                       ])
    logreg.fit(X_train, y_train)

    # % % time
    y_pred = logreg.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    # print(df.head(5))
    # print(df['post'][5])
    # clean up the text
    # df['post'] = df['post'].apply(clean_text)
    # print(df["post"])
    # print(df['post'].apply(lambda x: len(x.split(' '))).sum())

    #1. Naive Bayes Classifier for Multinomial Models using scikit
    naiveBayesClassifier(df, X_train, X_test, y_train, y_test)

    #2. Linear Support Vector Machine
    SVM(df, X_train, X_test, y_train, y_test)
    #

    #3. Logistic Regression
    logisticRegression(df, X_train, X_test, y_train, y_test)

    #4. Word2vec and Logistic  Regression
    w2v = Word2VecLR(df)
    w2v.execute()

    #5. Doc2vec and Logistic Regression
    d2v = Doc2VecLR(df)
    d2v.execute()

    #6. BOW with Keras
    bwKeras = BOWKeras(df)
    bwKeras.execute()