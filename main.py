from model import Model
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')


#model = Model("raw_data_back.csv")
#model.save("our_model.pickle")
loaded_model = Model.load("our_model1.pickle")
data=loaded_model.data_for_model
df2=loaded_model.final_groups.final_df
data = data[['overall', 'ordinal' , 'reviewText','label']]
#g.to_csv('./raw_data_back_model.csv', sep = '\t',index=False)
'''import matplotlib.pyplot as plt
import seaborn as sns
plt.bar('AVG',df2.loc[:,'avg_rating_deviation'],)
plt.bar('GD',df2.loc[:,'group_deviation']) 
plt.bar('GS',df2.loc[:,'group_size'])
plt.bar('RT',df2.loc[:,'review_tightness'])
plt.bar('BS',df2.loc[:,'burst_ratio'])
plt.bar('GS',df2.loc[:,'group_support'])
plt.bar('GSR',df2.loc[:,'group_size_ratio'])
plt.bar('GCS',df2.loc[:,'group_content_similarity'])
plt.ylim(0,4) 
plt.title('Bar graph') 
plt.ylabel('Suspicious Score between 0 and 1') 
plt.xlabel('Behavioural Features with k=7')
sns.barplot(x=['group_deviation','group_size'],y=['suspicious_score'],data=df1)'''


'''lemmatizer = WordNetLemmatizer()
corpus = []
for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['reviewText'][i])
    review = review.lower()
    review = review.split()
    
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

import pickle
with open('corpus1.pkl','wb') as f:
    pickle.dump(corpus,f)
with open('corpus1.pkl','rb') as f:
    corpus=pickle.load(f)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=8000)
X = cv.fit_transform(corpus).toarray()

y=data.iloc[:,3]

# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)
y_pred_train=spam_detect_model.predict(X_train)

from sklearn.metrics import confusion_matrix
conf=confusion_matrix(y_test,y_pred)
conf_train=confusion_matrix(y_train,y_pred_train)

from sklearn.metrics import accuracy_score
testing_accuracy=accuracy_score(y_test,y_pred)
training_accuracy=accuracy_score(y_train,y_pred_train)


import xgboost
classifier = xgboost.XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()'''

    