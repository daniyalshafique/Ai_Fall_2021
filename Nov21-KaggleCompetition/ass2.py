import pandas as pd
test = pd.read_csv('test.csv')

import random as rand
id = test["id"]
Test = pd.DataFrame(id)
arr=[]
for row in id:
  arr.append(rand.randint(0,1))
Test["target"] = arr
Test.to_csv('areeb.csv',index = False)

print(Test)