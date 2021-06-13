import pandas as pd
import pickle

df = pd.read_csv('D:/Jayant2/Deployment/spam_ham/spam.csv',encoding = 'iso-8859-1')
df.head()

df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis = 1)

df = df.rename(columns = {'v2':'messages','v1':'labels'})

df.head()


# Text Preprocessing
df.isna().sum()

df['labels'].value_counts()

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import re
stop = set(stopwords.words('english'))
def clean_text(text):
  text = text.lower()
  text = re.sub(r'[^0-9a-zA-Z]',' ',text)
  text = re.sub(r'\s+',' ',text)
  text = ' '.join(word for word in text.split() if word not in stop)
  return text

df['messages'] = df['messages'].apply(clean_text)

df.head()

X = df['messages']
y = df['labels']

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X = cv.fit_transform(X)

pickle.dump(cv,open('transform.pkl','wb'))

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(X_train,y_train)

model.score(X_test,y_test)

filename = 'nlp_model.pkl'
pickle.dump(model,open(filename,'wb'))

