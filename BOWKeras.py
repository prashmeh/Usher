# % matplotlib inline
import numpy as np
from keras import utils
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.preprocessing import text
from sklearn.preprocessing import LabelEncoder

class BOWKeras:
    df = ""
    num_classes = 0
    batch_size = 32
    max_words = 1000

    def __init__(self, df):
        self.df = df

    def execute(self):
        train_size = int(len(self.df) * .7)
        train_posts = self.df['post'][:train_size]
        train_tags = self.df['tags'][:train_size]

        test_posts = self.df['post'][train_size:]
        test_tags = self.df['tags'][train_size:]
        encoder = LabelEncoder()
        tokenize = text.Tokenizer(num_words=self.max_words, char_level=False)
        model = self.trainAndBuildModel(train_posts, train_tags, tokenize, encoder)

        x_test = tokenize.texts_to_matrix(test_posts)
        y_test = encoder.transform(test_tags)
        y_test = utils.to_categorical(y_test, self.num_classes)
        self.testTheBuildModel(model, x_test, y_test)

    def trainAndBuildModel(self, train_posts, train_tags, tokenize, encoder):
        tokenize.fit_on_texts(train_posts)  # only fit on train

        x_train = tokenize.texts_to_matrix(train_posts)

        encoder.fit(train_tags)
        y_train = encoder.transform(train_tags)

        num_classes = np.max(y_train) + 1
        y_train = utils.to_categorical(y_train, num_classes)

        epochs = 2

        # Build the model
        model = Sequential()
        model.add(Dense(512, input_shape=(self.max_words,)))
        model.add(Activation('relu'))
        # model.add(Dropout(0.5))  #We had to comment this out as it was throwing Runtime error....
        # Now it may lead to overfitting. The issue can be due to incompatibility b/w
        # python and TF libs

        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy']
                      )

        history = model.fit(x_train, y_train,
                            batch_size=self.batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_split=0.1)
        return model


    def testTheBuildModel(self, model, x_test, y_test):
        # Accuracy
        score = model.evaluate(x_test, y_test, batch_size=self.batch_size, verbose=1)
        print('Test accuracy:', score[1])
