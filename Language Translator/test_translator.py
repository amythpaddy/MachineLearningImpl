from pickle import load

from keras.utils import to_categorical
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


# load a clean dataset
def load_doc(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text

def to_sentences(doc):
    return doc.lower().strip().split('\n')

def to_pairs(text):
    listss = text.strip().split('\n')
    X = list()
    y = list()
    for lists in listss:
        a=lists.split('  ')
        X.append(a[0])
        y.append(a[1])
    return X,y

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    print(prediction)
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)




def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    return max(len(line.split()) for line in lines)

def encode_sequence(tokenizer, lenth, lines):
    X = tokenizer.texts_to_sequences(lines)
    X = pad_sequences(X, maxlen=lenth, padding='post')
    return X

def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y.reshape(sequences.shape[0],sequences.shape[1],vocab_size)
    return y



# text = load_doc('dataset.txt')
# X,y = to_pairs(text)
X = load_doc('X.txt')
X = to_sentences(X)

y = load_doc('y.txt')
y = to_sentences(y)

sentence = ["I declare resumed the session of the European Parliament adjourned on Friday 17 December 1999, and I would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive period."]

hin_tokenizer = create_tokenizer(y)
eng_tokenizer = create_tokenizer(X)
hin_vocab_size = len(hin_tokenizer.word_index)+1
hin_length = max_length(y)
eng_length = max_length(X)

model = load_model("model.h5")
test_tokenizer = create_tokenizer(sentence)

trainX = encode_sequence(eng_tokenizer, eng_length, sentence)

# predict = evaluate_model(model, trainX)
predict = predict_sequence(model, hin_tokenizer,trainX)
print(predict)