import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC

import sqlalchemy
from sqlalchemy import *

print("connecting to postgres database")
engine = sqlalchemy.create_engine('postgresql+psycopg2://postgres:r3dw00d@localhost/youtube_trends')

print("reading in sql table")
data = pd.read_sql_table(table_name="yt_trending_videos", con=engine)

print("formatting data")
df = data[["title","tags","description","category_id"]].copy()
#Brief look at all the text we can work with. We will be looking to combine these and clean the data as much as possible.
#From a brief analysis, the descriptions of the videos do not tend to describe much of anything to its irony. It is mostly used to further branding and social media presence.
df["tags"] = df["tags"].apply(lambda x: " " + x)
df["corpus"] = df["title"] + df["tags"]
df.drop(["title","tags","description"], axis=1, inplace=True)
df["corpus"] = df["corpus"].apply(lambda x: x.replace("|"," ").replace("\"", ""))
df.rename(columns={"category_id":"category"}, inplace=True)
#Assign numerical values to distinct items
df['_id'] = df['category'].factorize()[0]
#Create a lookup table based on the unique categories
category_id_df = df[['category', '_id']].drop_duplicates().sort_values('_id')
#Create a dictionary for easy manipulation
category_to_id = dict(category_id_df.values)
#Inverted lookup table
id_to_category = dict(category_id_df[['_id', 'category']].values)
#Lets find the Term Frequency Inverse Document Frequency 
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.corpus).toarray()
labels = df._id


print("training updated data")
model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
clf = model.fit(X_train, y_train)

text = [input("Test titles or tags then press Enter: ")]
text_features = tfidf.transform(text)
predictions = model.predict(text_features)
for text, predicted in zip(text, predictions):
    print('"{}"'.format(text))
    print("  - Predicted as: '{}'".format(id_to_category[predicted]))
    print("")

res = ["hold"]
while res[0] != 'q':
    res = [input("Try again or press 'Q + Enter' to exit: ")]
    text = res
    text_features = tfidf.transform(text)
    predictions = model.predict(text_features)
    for text, predicted in zip(text, predictions):
        print('"{}"'.format(text))
        print("  - Predicted as: '{}'".format(id_to_category[predicted]))
        print("")
    
else:
    print("Thanks for using this Machine Learning model!")