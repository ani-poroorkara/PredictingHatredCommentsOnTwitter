# Predicting Hatred Comments On Twitter
Predict Hatred tweets from twitter using the dataset [Twitter Sentiment Analysis](https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech).

## Working of the Model
This model uses deep convolutional layers with multichannel input layers. 
### Steps:
#### Pre-processing 
1. Load data as dataframe.
2. Drop unrequired columns.
3. Remove usernames from tweets.
4. Remove #s.
5. Remove emojies as it will not be recognisable by normal character.
#### Tokenizer
1. Fit the train data on a tokeniser to convert all the strings into numbers.
2. Encode the strings in the dataset into numbers.
#### Padding
The ends of the sentences are padded with 0 (zero) to indicate the end of the string.
#### Model 
We use 1DConvolution layers for text classification.
1. The first channel has a kernal size of 2. 
2. The second channel has a kernal size of 3.
3. The third channel has a kernal size of 4.
4. All outputs are concatenated. 
5. Dense layer is added.
6. Dropout layer is added. 
7. Final output dense layer with one output class is added.

## Dependencies
```python
import pandas as pd
import numpy as np
import re
import tensorflow_datasets as tfds
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences as pad_seq
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
import statistics
```

## Run the model
1. Clone the repository 
2. Install dependencies
3. Run the sentimentAnalysis.py 

### License
[Apache License 2.0](https://github.com/ani-poroorkara/PredictingHatredCommentsOnTwitter/blob/master/LICENSE)

##### I recommend using Google Colab or Jupyter notebooks to run the file cell by cell
##### Connect with me on [LinkedIn](https://www.linkedin.com/in/anirudh-poroorkara-34900017b/)
