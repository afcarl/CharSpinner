
# coding: utf-8

# In[106]:


#https://medium.com/@curiousily/making-a-predictive-keyboard-using-recurrent-neural-networks-tensorflow-for-hackers-part-v-3f238d824218
# note: I trained on alice in wonderland, about 1/4 the length of his Nietzsche data set.
# I also tested with "seeds" that come from his Nietzsche examples!
# TODO history
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


get_ipython().magic('matplotlib inline')
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 12, 5


# In[107]:


trainP=True
useNietzsche=False
useCarroll=False
useShakespeare=True

LOGDIR="LOG"
RUNNAME="minishakespeareStep3Seq50"
FILENAME=LOGDIR + "/" + RUNNAME

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

SEQUENCE_LENGTH = 50
EPOCHS=15
step = 3   #skip this number of chars for generating new training sequences
layer1size=128
topN=1
topNStartWord=3
k_phraseLength=100

k_layers=2
k_bn=True
k_batchsize=128
k_lr=.005

k_stateful=False
k_shuffle=True
if (k_stateful) :
    k_shuffle=False
    
k_validationSplit=.1
    
k_condNietzsche=[1,0,0]
k_condCarroll=[0,1,0]
k_condShakespeare=[0,0,1]
lenconditional=len(k_condNietzsche) # the are all the same length, of course


# In[108]:


import re as re
def cleanText(text) :
    # replace all numbers followed by an optional letter and then a dot (eg numbered paragraphs)                                    
    text = re.sub("(^|\W)\d+[a-zA-Z]*($|\W|\.)", "", text)
    #escaped apotrophes                                                                                                             
    #text = text.replace('\n', ' ').replace("\'", "'").replace("\"","").replace('[Illustration]',"").replace('*',"")
    text = text.replace("\'", "'").replace("\"","").replace('[Illustration]',"").replace('*',"")
    #repeated white space
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
text3=""

if useNietzsche :
    path = 'nietzsche.txt'
    text1 = cleanText(open(path, 'r', encoding='utf-8').read().lower())
    print('corpus 1 length:', len(text1))
    print("NIETZSCHE CHARS: ", sorted(list(set(text1))))
    
if useCarroll :
    path = 'carroll.txt'
    text2 = cleanText(open(path, 'r', encoding='utf-8').read().lower())
    print('corpus 2 length:', len(text2))
    print("CARROLL CHARS: ", sorted(list(set(text2))))

if useShakespeare :
    path = 'minishakespeare.txt'
    text3 = cleanText(open(path, 'r', encoding='utf-8').read().lower())
    print('corpus 3 length:', len(text3))
    print("SHAKESPEARE CHARS: ", sorted(list(set(text3))))
    
text=text1+text2+text3
print('total cleaned corpus length is ', len(text))

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

lenchars=len(chars)
lenAugmentedInput=lenchars+lenconditional

#print(f'unique chars: {len(chars)}')
print('unique chars: ', str(len(chars)))
#chars
#indices_char
#text


# In[109]:


#CREAT TRAINING DATA
# cut the corpus into chunks of 40 characters, spacing the sequences by 3 characters
# Additionally, we will store the next character (the one we need to predict) for every sequence

sentences = []
next_chars = []
cond_input=[]

#grab as many full batches as possible, ignoring partial batch left over
samples1= (int(len(text1)/step) - SEQUENCE_LENGTH)-(int(len(text1)/step) - SEQUENCE_LENGTH)%k_batchsize
samples2= (int(len(text2)/step) - SEQUENCE_LENGTH)-(int(len(text2)/step) - SEQUENCE_LENGTH)%k_batchsize
samples3= (int(len(text3)/step) - SEQUENCE_LENGTH)-(int(len(text3)/step) - SEQUENCE_LENGTH)%k_batchsize

if useNietzsche :
    for i in range(0, samples1*step, step):
        sentences.append(text1[i: i + SEQUENCE_LENGTH])
        next_chars.append(text1[i + SEQUENCE_LENGTH])
        cond_input.append(k_condNietzsche)

if useCarroll :
    for j in range(0, samples2*step, step):
        sentences.append(text2[j: j + SEQUENCE_LENGTH])
        next_chars.append(text2[j + SEQUENCE_LENGTH])
        cond_input.append(k_condCarroll)

if useShakespeare :
    for k in range(0, samples3*step, step):
        sentences.append(text3[k: k + SEQUENCE_LENGTH])
        next_chars.append(text3[k + SEQUENCE_LENGTH])
        cond_input.append(k_condShakespeare)


#print(f'num training examples: {len(sentences)}')
print('num training examples:  ', str(len(sentences)))
print('num batches:  ', str(len(sentences)/k_batchsize))

if (True) : # (k_stateful) : # ALWAYS do full batches for training and testing (required for k_stateful, anyway)
    # adjust the validation split so that it has an integer number of batches of size k_batchsize
    numvexamples=k_validationSplit*len(sentences)  #target number
    numvexamples=numvexamples-numvexamples%k_batchsize #divisible by batch size
    k_validationSplit=numvexamples/len(sentences) #adjusted split number for fit()

    print('num validation examples:  ', str(numvexamples))
    print('k_validationSplit:  ', str(k_validationSplit))


# In[110]:


# save parameters of run
param={'FILENAME': FILENAME, 'RUNNAME': RUNNAME, 'SEQUENCE_LENGTH': SEQUENCE_LENGTH, 'EPOCHS': EPOCHS, 'step': step, 'layer1size': layer1size, 'k_layers': k_layers, 'k_bn': k_bn, 'k_batchsize': k_batchsize, 'k_lr': k_lr, 'k_stateful': k_stateful, 'k_shuffle': k_shuffle, 'k_validationSplit': k_validationSplit, 'chars': chars}  

with open(FILENAME + '.params.pkl', 'wb') as f:  
    pickle.dump(param, f)
    
#CHECK: get parameters from training param file
with open(FILENAME + '.params.pkl', 'rb') as f:  
    param = pickle.load(f)
# and PRINT
for key, value in param.items():
    print(key,  " : ", value)


# In[111]:


# generate features and labels - one-host versions of the input and prediction vectors

X = np.zeros((len(sentences), SEQUENCE_LENGTH, lenAugmentedInput), dtype=np.bool)  #x[sample_index][one-hot array]
y = np.zeros((len(sentences), lenchars), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
        X[i, t,-lenconditional:] = cond_input[i] # set conditional bit on last charcter in each training sample
    y[i, char_indices[next_chars[i]]] = 1


# In[112]:


k_validationSplit*len(y)/128


# In[113]:



print("len of first input vector, first character vector x is ", str(len(X[0][0])))
print("a character in a sentice is ", X[1410][0] )

print(str(X.shape))   #training_samples, SEQUENCE_LENGTH, lenAugmentedInput


# In[114]:


for i in range(k_layers) :
    print( str(i))


# In[115]:


#LST layer with 128 neurons
# takes a shape which is 
model = Sequential()

for i in range(k_layers) :
    #if last layer, don't return sequence
    if i==(k_layers-1) :
        model.add(LSTM(layer1size,  batch_size=k_batchsize, stateful=k_stateful, input_shape=(SEQUENCE_LENGTH, lenAugmentedInput)))
    else :
        model.add(LSTM(layer1size,   batch_size=k_batchsize, stateful=k_stateful, return_sequences=True, input_shape=(SEQUENCE_LENGTH, lenAugmentedInput)))
        
    #FAIL model.add(LSTM(layer1size, dropout=0.25, recurrent_dropout=0.25, input_shape=(SEQUENCE_LENGTH, lenAugmentedInput)))
    #FAIL model.add(GRU(layer1size,  input_shape=(SEQUENCE_LENGTH, lenAugmentedInput), kernel_regularizer=regularizers.l2(0.1), recurrent_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1), activity_regularizer=regularizers.l2(0.1)))

    if (k_bn) : 
        model.add(BatchNormalization())

#model.add(Dense(lenchars, kernel_regularizer=regularizers.l2(0.1),  bias_regularizer=regularizers.l2(0.1), activity_regularizer=regularizers.l2(0.1)))
#model.add(Dense(lenchars, kernel_regularizer=regularizers.l2(0.1),  bias_regularizer=regularizers.l2(0.1), activity_regularizer=regularizers.l2(0.1)))
model.add(Dense(lenchars))
model.add(Activation('softmax'))


# In[116]:


# Train. Validate with 5% of the examples

if trainP :
    optimizer = RMSprop(lr=k_lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # checkpoint
    filepath= FILENAME + ".cp-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1)
    callbacks_list = [checkpoint]


    history = model.fit(X, y, validation_split=k_validationSplit, batch_size=k_batchsize, epochs=EPOCHS, shuffle=k_shuffle, callbacks=callbacks_list).history

    #Save (How does this work ??)
    model.save(FILENAME + '.keras_model.h5')
    pickle.dump(history, open(FILENAME + '.history.p', 'wb'))

