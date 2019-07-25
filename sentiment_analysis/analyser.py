import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
#import matplotlib.pyplot as plt
import re


data = pd.read_csv('train.tsv',sep='\t')
data = data[['Phrase','Sentiment']]
print(data.head(5))
data['Phrase'] = data.Phrase.astype(str)

data = data[data.Sentiment != 2]

data['Phrase'] = data['Phrase'].apply(lambda x: x.lower())
data['Phrase'] = data['Phrase'].apply((lambda x: re.sub('[^a-zA-Z0-9\s]','',x)))

data = data[data['Sentiment'] != 2]

print(data[ data['Sentiment'] == 0].size + data[data['Sentiment'] == 1].size)
print(data[ data['Sentiment'] == 3].size+data[ data['Sentiment'] == 4].size)

# for idx, row in data.iterrows():
#     row[0] = row[0].replace('rt',' ')

max_features = 5000
tokenizer = Tokenizer(num_words=max_features, split=' ')
X = tokenizer.fit_on_texts(data['Phrase'].values)
X = tokenizer.texts_to_sequences(data.Phrase.values)
print(X)
X= pad_sequences(X)

embed_dim = 128
lstm_out=196
print(X.shape)

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['Sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state=42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
#
batch_size = 32
history = model.fit(X_train, Y_train, epochs=20, batch_size=batch_size, verbose=2)
model.save('sentiment_analysis.h5')
#
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

validation_size=1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]

score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))


twt = ['is this not good']
twt = tokenizer.texts_to_sequences(twt)
twt = pad_sequences(twt,maxlen=46, dtype='int32', value=0)
print(twt)

sentiment = model.predict(twt, batch_size=1, verbose=2)[0]
if(sentiment[0]+sentiment[1]>sentiment[2]+sentiment[3]):
    print('negetive')
else:
    print('positive')