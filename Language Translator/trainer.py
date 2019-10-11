import pandas as pd
from keras import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Embedding, RepeatVector
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


data = pd.read_csv('data.csv')
eng = data['english']
hin = data['hindi']


def create_tokenizer(lang_data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lang_data)
    return tokenizer


def max_length(lang_data):
    return max(len(line.split()) for line in lang_data)


def encode_sequence(tokenizer, lang_data):
    seq = tokenizer.texts_to_sequences(lang_data)
    seq = pad_sequences(seq, maxlen=max_length(lang_data), padding="post")
    return seq


eng_token = create_tokenizer(eng)
eng_vocab_len = len(eng_token.word_index) + 1
eng_max_len = max_length(eng)
eng_seq = encode_sequence(eng_token, eng)

hin_token = create_tokenizer(hin)
hin_vocab_len = len(hin_token.word_index) + 1
hin_max_len = max_length(hin)
hin_seq = encode_sequence(hin_token, hin)

print(eng_seq.shape)
print(hin_seq.shape)

eng_timestep = 1
hin_timestep = 1
# eng_seq = eng_seq.reshape(len(eng), eng_timestep, eng_max_len)
hin_seq = hin_seq.reshape(len(hin), hin_timestep, hin_max_len)
print(eng_seq.shape)
print(hin_seq.shape)


def lstm_model():
    model = Sequential()
    model.add(Embedding(eng_vocab_len,10,input_length=eng_max_len))
    model.add(LSTM(100))
    model.add(RepeatVector(hin_timestep))
    # model.add(LSTM(len(eng), input_shape=(eng_timestep, eng_max_len), return_sequences=True))
    model.add(LSTM(100,return_sequences=True))
    model.add(TimeDistributed(Dense(hin_max_len)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model


def word_for_id(id, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == id:
            return word
    return None


def get_sequence(tokenizer, results):
    result = [round(vector) for vector in results]
    target = list()
    for i in result:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)


model = lstm_model()
model.fit(eng_seq, hin_seq, epochs=2000, batch_size=5)
model.save("eng_to_hin.h5")