import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils
from sklearn import metrics

# READING CSV FILE
train_data = pd.read_csv('/train.csv')
test_data = pd.read_csv('/test.csv')

test_data.head(2)

train_data.head(2)

train = pd.DataFrame(train_data)
train_norm=(train-train.min())/(train.max()-train.min())
X = train_norm.Elevation
Y = train_norm.Cover_Type
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=255)
print("X_train : ",len(X_train))
print("X_test : ",len(X_test))
print("X_train : ",len(y_train))
print("X_test : ",len(y_test))
# TRAINING MODEL
model =LogisticRegression(solver = 'newton-cg')
scaler = MinMaxScaler()
lab_enc = preprocessing.LabelEncoder()
X_train = lab_enc.fit_transform(X_train)
A= X_train.reshape(-1,1)
X_train = scaler.fit_transform(A)
y_train = lab_enc.fit_transform(y_train)
Test = test_data.values.reshape(-1,1)
X_test = scaler.fit_transform(Test)
model.fit(X_train,y_train)
L = model.predict(Test)
# Model (LG_NORM)Accuracy
print("Accuracy:",metrics.accuracy_score(Test, L))

id = test_data["Id"]
Test = pd.DataFrame(id)
arr=[]
for row in id:
  arr.append(L[row])
Test["Cover type"] = arr
Test.to_csv('Score.csv',index = False)