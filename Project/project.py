# -*- coding: utf-8 -*-
"""Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-9I5PuBLhbLKMHd9xAc3TsQmyVJ6v3ey
"""

# -*- coding: utf-8 -*-
"""Assign02-Naive Bayes(1)

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MlyvGkT4dfNreOQ95GlRz3ceZOklXjUa
"""

from google.colab import drive
# This will prompt for authorization.
drive.mount('/content/drive')

import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 	
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

train_data = pd.read_csv('/content/drive/MyDrive/train.csv')

train_data.head(5)

train = pd.DataFrame(train_data)

y = train.Cover_Type
X = train.drop("Cover_Type", axis=1)

t_train, t_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109)
print(t_train.head())
print(t_test.head())