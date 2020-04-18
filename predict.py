import pandas as pd
import numpy as np
from keras.models import Model
import statistics
import pickle

#Load the model
SA_model = load_model('model/SA_model.h5')

def predict_class_avg(prediction):
  avg_class = []
  avg = sum(prediction) / len(prediction)
  for value in prediction:
    if value > avg:
      avg_class.append(1)
    else:
      avg_class.append(0)
  return avg_class
  
def predict_class_med(prediction):
  med_class = []
  med = statistics.median(prediction)
  for value in prediction:
    if value > med:
      med_class.append(1)
    else:
      med_class.append(0)
  return med_class


with open("files/encoded_test.txt", "rb") as fp:
  encodedData_test = pickle.load(fp)

prediction = SA_model.predict([encodedData_test,encodedData_test,encodedData_test])


typeClass = input("Input 1 for average and 2 for median")

if int(typeClass) == 1:
  result = predict_class_avg(prediction)
elif int(typeClass) == 2:
  result = predict_class_med(prediction)

print(result) 
