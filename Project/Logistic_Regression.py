#code by AREEB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# SPLITTING AND DISTRIBUTING DATA ON X AND Y AXIS
train = pd.read_csv('/train.csv')
test = pd.read_csv('/test.csv')

test_data.head(2)

train_data.head(2)
X = train.Elevation
Y = train.Cover_Type
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=109)
print("X_train : ",len(X_train))
print("X_test : ",len(X_test))
print("X_train : ",len(y_train))
print("X_test : ",len(y_test))
# TRAINING MODEL
model =LogisticRegression(solver = 'newton-cg')
scaler = MinMaxScaler()
A= X_train.values.reshape(-1,1)
X_train = scaler.fit_transform(A)
Test = test_data.values.reshape(-1,1)
X_test = scaler.fit_transform(Test)
model.fit(X_train,y_train)
L = model.predict(Test)	
# Model (Logistic Regression) Accuracy
print("Accuracy:",metrics.accuracy_score(Test, L))

id = test_data["Id"]
Test = pd.DataFrame(id)
arr=[]
for row in id:
  arr.append(L[row])
Test["Cover type"] = arr
Test.to_csv('score.csv',index = False)
