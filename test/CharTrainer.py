
# coding: utf-8

# In[20]:


#https://medium.com/@curiousily/making-a-predictive-keyboard-using-recurrent-neural-networks-tensorflow-for-hackers-part-v-3f238d824218
# note: I trained on alice in wonderland, about 1/4 the length of his Nietzsche data set.
# I also tested with "seeds" that come from his Nietzsche examples!
# TODO 
#    - what happens if you let it run further by itself? 
#   - Use  GRU instead of LSTM

import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, Dropout,BatchNormalization
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
from keras import regularizers
import matplotlib.pyplot as plt
import pickle
import sys
import os

from keras.callbacks import ModelCheckpoint

import heapq
import seaborn as sns
from pylab import rcParams




trainP=True
useNietzsche=True
useCarroll=True

#FILENAME="carroll.3.20"   #for writing or reading
#FILENAME="nietzsche.3.20"   #for writing or reading
LOGDIR="LOG"
FILENAME=LOGDIR + "/foobar"

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

SEQUENCE_LENGTH = 40
EPOCHS=1
step = 3   #skip this number of chars for generating new training sequences
layer1size=128
topN=1
topNStartWord=3
k_phraseLength=100

k_condNietzsche=[1,0]
k_condCarroll=[0,1]


# In[22]:


import re as re
def cleanText(text) :
    # replace all numbers followed by an optional letter and then a dot (eg numbered paragraphs)                                    
    text = re.sub("(^|\W)\d+[a-zA-Z]*($|\W|\.)", "", text)
    #escaped apotrophes                                                                                                             
    text = text.replace('\n', ' ').replace("\'", "'").replace("\"","").replace('[Illustration]',"").replace('*',"")
    #repeated white space
    text=re.sub('\s{2,}',' ', text)
    text=re.sub('â', 'a', text)
    text=re.sub('æ', 'a', text)
    text=re.sub('è', 'e', text)
    text=re.sub('ï', 'i', text)
    text=re.sub('ù', 'u', text)
    text=re.sub('&c', 'etc', text)
    text=re.sub('\ufeff', '', text)
    text=re.sub('‘', "'", text)
    text=re.sub('’', "'", text)
    text=re.sub('“', "'", text)
    text=re.sub('”', "'", text)

    # try to normalize Carroll text a bit more, although there are still way more contractions in carrll than nietzsche             
    text=re.sub('!', " ", text)
    text=re.sub('\?', " ", text)
#                                                                                                                                   
    text=re.sub('--', " ", text)
    text=re.sub('_', " ", text)

    #repeated white space                                                                                                           
    text=re.sub('\s{2,}',' ', text)

    return text

text1=""
text2=""

if useNietzsche :
    path = 'nietzsche.txt'
    text1 = cleanText(open(path, 'r', encoding='utf-8').read().lower())
    print('corpus 1 length:', len(text1))
    print("NIETZXCHE CHARS: ", sorted(list(set(text1))))
    
if useCarroll :
    path = 'carroll.txt'
    text2 = cleanText(open(path, 'r', encoding='utf-8').read().lower())
    print('corpus 2 length:', len(text2))
    print("CARROLL CHARS: ", sorted(list(set(text2))))

text=text1+text2
print('total cleaned corpus length is ', len(text))

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

lenchars=len(chars)
lenconditional=2
lenAugmentedInput=lenchars+lenconditional

#print(f'unique chars: {len(chars)}')
print('unique chars: ', str(len(chars)))
#chars
#indices_char



#CREAT TRAINING DATA
# cut the corpus into chunks of 40 characters, spacing the sequences by 3 characters
# Additionally, we will store the next character (the one we need to predict) for every sequence

sentences = []
next_chars = []
cond_input=[]

if useNietzsche :
    for i in range(0, len(text1) - SEQUENCE_LENGTH, step):
        sentences.append(text1[i: i + SEQUENCE_LENGTH])
        next_chars.append(text1[i + SEQUENCE_LENGTH])
        cond_input.append(k_condNietzsche)

if useCarroll :
    for j in range(0, len(text2) - SEQUENCE_LENGTH, step):
        sentences.append(text2[j: j + SEQUENCE_LENGTH])
        next_chars.append(text2[j + SEQUENCE_LENGTH])
        cond_input.append(k_condCarroll)

#print(f'num training examples: {len(sentences)}')
print('num training examples:  ', str(len(sentences)))


# In[26]:


# generate features and labels - one-host versions of the input and prediction vectors

X = np.zeros((len(sentences), SEQUENCE_LENGTH, lenAugmentedInput), dtype=np.bool)  #x[sample_index][one-hot array]
y = np.zeros((len(sentences), lenchars), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
        X[i, t,-lenconditional:] = cond_input[i] # set conditional bit on last charcter in each training sample
    y[i, char_indices[next_chars[i]]] = 1


# In[27]:



print("len of first input vector, first character vector x is ", str(len(X[0][0])))
print("a character in a sentice is ", X[141000][0] )


# In[28]:


print(str(X.shape))   #training_samples, SEQUENCE_LENGTH, lenAugmentedInput


# In[29]:


#LST layer with 128 neurons
# takes a shape which is 
model = Sequential()
model.add(LSTM(layer1size,  input_shape=(SEQUENCE_LENGTH, lenAugmentedInput)))

#FAIL model.add(LSTM(layer1size, dropout=0.25, recurrent_dropout=0.25, input_shape=(SEQUENCE_LENGTH, lenAugmentedInput)))
#FAIL model.add(GRU(layer1size,  input_shape=(SEQUENCE_LENGTH, lenAugmentedInput), kernel_regularizer=regularizers.l2(0.1), recurrent_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1), activity_regularizer=regularizers.l2(0.1)))

#model.add(Dense(lenchars, kernel_regularizer=regularizers.l2(0.1),  bias_regularizer=regularizers.l2(0.1), activity_regularizer=regularizers.l2(0.1)))
#model.add(Dense(lenchars, kernel_regularizer=regularizers.l2(0.1),  bias_regularizer=regularizers.l2(0.1), activity_regularizer=regularizers.l2(0.1)))
model.add(BatchNormalization())
#model.add(Dropout(0.5))
model.add(Dense(lenchars))

model.add(Activation('softmax'))


# In[30]:


FILENAME + '.history.p'


# In[31]:


# Train. Validate with 5% of the examples


if trainP :
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # checkpoint
    filepath= FILENAME + ".cp-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1)
    callbacks_list = [checkpoint]


    history = model.fit(X, y, validation_split=0.1, batch_size=128, epochs=EPOCHS, shuffle=True, callbacks=callbacks_list).history

    #Save (How does this work ??)
    model.save(FILENAME + '.keras_model.h5')
    pickle.dump(history, open(FILENAME + 'history.p', 'wb'))


# In[32]:

