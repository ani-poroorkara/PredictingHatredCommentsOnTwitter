#!unzip twitter-sentiment-analysis-hatred-speech.zip


import pandas as pd
import numpy as np

#Load test and train data
#Note that test data does not contain any labels 
df =  pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df.drop('id', axis=1,inplace=True)
df_test.drop('id', axis=1,inplace=True)


import re

#Define a corpus from which learning will take place 
corpus = []
#Define a different list to store the labels
labellist = []

#Define a test for the unknown tweets
test = []

#This is a function to remove emojis in ascii 
def deEmojify(inputString):
  return inputString.encode('ascii', 'ignore').decode('ascii')

#Removing the mentioned "impurities" from the tweet and to clean the data
#testData variable is used so that we can differentiate between test data and train data
def process_tweet(tweet_text, label, testData = False):
  tweet_text = re.sub(r"@user", "", tweet_text)
  tweet_text = re.sub(r"@", "", tweet_text)
  tweet_text = re.sub(r"#", "", tweet_text)
  tweet_text = deEmojify(tweet_text)
  tweet_text = re.sub(r" +", " ", tweet_text)
  if testData == False:
    corpus.append(tweet_text)
    labellist.append(label)
  elif testData == True:
    test.append(tweet_text)
  print("#"*10, end='')

#Define count of number of train data text
count = 0
print("<-"*5 + "TRAIN DATA CORPUS AND LABELS" + "->"*5)
for tweet in df.iterrows():
  label = tweet[1][0]
  txt = tweet[1][1] 
  process_tweet(txt, label)
  count += 1
  print(count)

#Define coutn of number of test data text
count_test = 0
print("<-"*5 + "TEST DATA CORPUS" + "->"*5)
for tweet in df_test.iterrows():
  txt = tweet[1][0] 
  process_tweet(txt, 0, True)
  count_test += 1
  print(count_test)
  

import tensorflow_datasets as tfds

#Create a tokeniser
#The tokensier converts each word into a number from a string
tokeniser = tfds.features.text.SubwordTextEncoder.build_from_corpus(corpus, target_vocab_size=2**16)

#Encode the strings into numbers
encodedData = []
for line in corpus:
  encodedData.append(tokeniser.encode(line))

encodedData_test = []
for line in test:
  encodedData_test.append(tokeniser.encode(line))
  

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences as pad_seq

#The ends of the sentences need to be padded so as to indicate different sentences 
#For this purpose we append zero (0) at the end of each sentence

maxlength = max([len(line) for sentence in encodedData])
encodedData = pad_seq(encodedData, value = 0, padding = 'post', maxlen = maxlength)

maxlengthtest = max([len(line) for sentence in encodedData_test])
encodedData_test = pad_seq(encodedData_test, value = 0, padding = 'post', maxlen = maxlengthtest)


from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate

#Define the multichannel model 
def define_model(length, vocab_size, emb_dim):
	#kernal size 2 
	inputs1 = Input(shape=(length,))
	embedding1 = Embedding(vocab_size, emb_dim)(inputs1)
	conv1 = Conv1D(filters=100, kernel_size=2, activation='relu')(embedding1)
	pool1 = MaxPooling1D(pool_size=2)(conv1)
	flat1 = Flatten()(pool1)
	
  #Kernal size 3
	inputs2 = Input(shape=(length,))
	embedding2 = Embedding(vocab_size, emb_dim)(inputs2)
	conv2 = Conv1D(filters=100, kernel_size=3, activation='relu')(embedding2)
	pool2 = MaxPooling1D(pool_size=2)(conv2)
	flat2 = Flatten()(pool2)
	
  #Kernal size 4
	inputs3 = Input(shape=(length,))
	embedding3 = Embedding(vocab_size, emb_dim)(inputs3)
	conv3 = Conv1D(filters=100, kernel_size=4, activation='relu')(embedding3)
	pool3 = MaxPooling1D(pool_size=2)(conv3)
	flat3 = Flatten()(pool3)
	
  #Merge all the outputs of the layers together 
	merged = concatenate([flat1, flat2, flat3])
	
  #Dense layers 
	dense = Dense(256, activation='relu')(merged)
	#Add a dropout layer to prevent overfitting 
	drop = Dropout(0.2)(dense)
	#Define the final out put layer 
	#We only have two outputs here 1 or 0 
	outputs = Dense(1, activation='sigmoid')(drop)
	
	#Define inputs and outputs of the model 
	model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
	
  #Compile the model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


#Create the model 
SA_model = define_model(maxlength, tokeniser.vocab_size, 200)

#Model summary 
SA_model.summary()

#Run the training
SA_model.fit([encodedData,encodedData,encodedData], labellist, epochs=5, batch_size=512)

#Save the model
SA_model.save('SA_model.h5')
