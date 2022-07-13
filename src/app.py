import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("data/processed/clean_bank.csv")

X = df.drop('y', axis=1)
y = df['y']

columns = X.columns
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_std = pd.DataFrame(X_std, columns = columns)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_std, y, test_size= 0.2, random_state=0)


model = LogisticRegression(C= 0.1, penalty='l2', solver= 'liblinear')
model.fit(X_train, y_train)

filename = 'models/model.sav'

pickle.dump(model, open(filename,'wb'))

#load model

loaded_model = pickle.load(open(filename,'rb'))

result = loaded_model.score(X_test, y_test)

print(result)

