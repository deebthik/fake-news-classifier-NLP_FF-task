

import os
import sys
import pandas

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from collections import Counter

import warnings
warnings.filterwarnings("ignore")



#---------------------| Loading the datasets |---------------------#

cwd = os.path.abspath(os.getcwd())

real_df = pandas.read_csv(cwd + "/True.csv")
fake_df = pandas.read_csv(cwd + "/Fake.csv")

print("\nNumber of real articles: " + str(len(real_df)))
print("Number of fake articles: " + str(len(fake_df)))


#---------------------Loading the datasets---------------------#



#---------------------| Preprocessing - Cleanup, Feature Extraction, Redundancy, Length, etc. |---------------------#

def clean_text(text):

  text = re.sub(r'\n', ' ', text)
  text = re.sub('\xa0', ' ', text)
  cleaned_text = re.sub('(http)(.+?)(?:\s)', ' ', text)

  return cleaned_text


df_list = [real_df, fake_df]

for df in df_list:
  df['text'] = df['text'].apply(lambda x: clean_text(x))


real_df["group"] = "real"
fake_df["group"] = "fake"

articles_df = real_df.append([fake_df], verify_integrity=True, ignore_index=True)


#Removing duplicates and redundancies

articles_df[articles_df["group"] == 'fake'] # of fake articles: 21226
len(articles_df[articles_df["group"] == 'fake'].text.unique()) # 16385 unique fake articles

articles_df[articles_df["group"] == 'real'] # of real articles: 20253
len(articles_df[articles_df["group"] == 'real'].text.unique()) # 20043 unique real articles

articles_df.drop_duplicates(subset=['text'], inplace=True, ignore_index=True)


articles_df['length'] = articles_df['text'].apply(lambda x: len(x))


#Removing short articles
articles_df.drop(articles_df[articles_df['length'] < 400].index, inplace=True)

articles_df.reset_index(drop=True, inplace=True)



def article_crop(article, char_limit):

  if len(article) > char_limit:

    before_limit = article[: char_limit]
    after_limit = article[char_limit : ]

    if len(nltk.sent_tokenize(after_limit)) > 0:

      sent_after_limit = nltk.sent_tokenize(after_limit)[0]

      new_text = before_limit + sent_after_limit
      return new_text

    else:
      return before_limit

  else:
    return article



article_crop(articles_df.text.iloc[63], 5711)[5400:]

articles_df['text'] = articles_df.text.apply(lambda x: article_crop(x, 5711))
articles_df['length'] = articles_df['text'].apply(lambda x: len(x))



#---------------------Preprocessing - Cleanup, Feature Extraction, Redundancy, Length, etc.---------------------#



#---------------------| Preprocessing - Metrics and Analysis |---------------------#


#Tokenizing sentences
articles_df['sentences'] = articles_df['text'].apply(lambda x: nltk.sent_tokenize(x))


def avg_words_per_sent(text):

  num_words_list = []

  for sent in range(len(text)):

    num_words = len([word for word in word_tokenize(text[sent]) if word.isalnum()])
    num_words_list.append(num_words)


  avg_word_per_sentences = sum(num_words_list) / len(num_words_list)

  return avg_word_per_sentences


articles_df['words_per_sent'] = articles_df['sentences'].apply(lambda x: avg_words_per_sent(x))



#Dropping articles with 100+ average words per sentence
articles_df.drop(articles_df[articles_df['words_per_sent'] > 100].index, inplace=True)

articles_df.reset_index(drop=True, inplace=True)

articles_df['text_lower'] = articles_df['text'].apply(lambda x: x.lower())

stopwords = set(stopwords.words('english'))


def tokenize_text(article):

  tok_text = nltk.word_tokenize(article)
  tok_text = [word.lower() for word in tok_text]
  tok_text = [word for word in tok_text if word not in stopwords]
  tok_text = [word for word in tok_text if word.isalpha()]

  return tok_text


#Creating a list of text articles for each group
real_text_list = list(articles_df.text[articles_df.group == "real"])
fake_text_list= list(articles_df.text[articles_df.group == "fake"])


real_word_list = [tokenize_text(article) for article in real_text_list]
fake_word_list = [tokenize_text(article) for article in fake_text_list]


#Flattening the list of lists into a single list of words
flat_real_word_list = [word for article in real_word_list for word in article]
flat_fake_word_list = [word for article in fake_word_list for word in article]


real_cnt = Counter(flat_real_word_list)
real_most_common = real_cnt.most_common(50)


fake_cnt = Counter(flat_fake_word_list)
fake_most_common = fake_cnt.most_common(50)



real_words = []
for i in range(len(real_most_common)):
  real_words.append(real_most_common[i][0])


fake_words = []
for i in range(len(fake_most_common)):
  fake_words.append(fake_most_common[i][0])


#Creating a set from the 50 most common words from both groups
human_words = set(real_words + fake_words)

#Tagging the most common words with parts of speech
nltk.pos_tag(human_words)

#Keeping only nouns out of all the words
tagged_human_words = nltk.pos_tag(human_words)

human_nouns = [word_tag_pair[0] for word_tag_pair in tagged_human_words if word_tag_pair[1] == 'NN']



def contains_word_in_list(text, word_list):

  if any(word in text for word in word_list):
    return True
  else:
    return False


articles_df["contains_human_word"] = articles_df['text_lower'].apply(lambda x: contains_word_in_list(x, human_nouns))
articles_df.drop(['contains_human_word'], axis=1, inplace=True)

articles_df.reset_index(drop=True, inplace=True)


#---------------------Preprocessing - Metrics and Analysis---------------------#




#---------------------| ML Model - Unsupervised -> Supervised |---------------------#


real_text = articles_df[articles_df['group'] == "real"][['text','group']][:1000]
fake_text = articles_df[articles_df['group']=='fake'][['text','group']][:500]

#Clustering
clustering_text_X = real_text['text'].append(fake_text['text'])

#Classification - Real vs. Fake
real_human_text_X = real_text['text'][:500].append(fake_text['text'])
real_human_text_y_raw = real_text['group'][:500].append(fake_text['group'])
real_human_text_y = []
for val in real_human_text_y_raw:
  if val == 'real':
    real_human_text_y.append(0)
  else:
    real_human_text_y.append(1)


#Classification - Real vs. Fake
real_allfake_text_X = real_text['text'].append(fake_text['text'])
real_allfake_text_y_raw = real_text['group'].append(fake_text['group'])
real_allfake_text_y = []
real_allfake_text_clusters = []
for val in real_allfake_text_y_raw:
  if val == 'real':
    real_allfake_text_clusters.append(1)
    real_allfake_text_y.append(0)
  elif val == 'fake':
    real_allfake_text_clusters.append(2)
    real_allfake_text_y.append(1)
  else:
    real_allfake_text_clusters.append(0)
    real_allfake_text_y.append(1)



#Gensim TF_IDF Vectorization
tf_idf_vectorizor2 = TfidfVectorizer()
tf_idf2 = tf_idf_vectorizor2.fit_transform(real_human_text_X)
tf_idf_norm2 = normalize(tf_idf2)
X_real_human = tf_idf_norm2.toarray()

tf_idf_vectorizor3 = TfidfVectorizer()
tf_idf3 = tf_idf_vectorizor3.fit_transform(real_allfake_text_X)
tf_idf_norm3 = normalize(tf_idf3)
X_real_allfake = tf_idf_norm3.toarray()


sklearn_pca = PCA(n_components = 2)
Y_sklearn = sklearn_pca.fit_transform(X_real_allfake)
kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')
fitted = kmeans.fit(Y_sklearn)
prediction = kmeans.predict(Y_sklearn)

clustering_df = pandas.DataFrame(Y_sklearn)
clustering_df.columns = ['x', 'y']
clustering_df['prediction'] = prediction


real_text = articles_df[articles_df['group'] == "real"][['sentences','group']][:16000]
fake_text = articles_df[articles_df['group']=='fake'][['sentences','group']][:16000]

#Clustering
clustering_text_X = real_text['sentences'].append(fake_text['sentences'])


#Classification - Real vs. Fake
real_human_text_X = real_text['sentences'].append(fake_text['sentences'])
real_human_text_y_raw = real_text['group'].append(fake_text['group'])
real_human_text_y = []
for val in real_human_text_y_raw:
  if val == 'real':
    real_human_text_y.append(0)
  else:
    real_human_text_y.append(1)


#Classification - Real vs. Fake
real_allfake_text_X = real_text['sentences'].append(fake_text['sentences'][:8000])
real_allfake_text_y_raw = real_text['group'].append(fake_text['group'][:8000])
real_allfake_text_y = []
real_allfake_text_clusters = []
for val in real_allfake_text_y_raw:
  if val == 'real':
    real_allfake_text_clusters.append(2)
    real_allfake_text_y.append(0)
  elif val == 'fake':
    real_allfake_text_clusters.append(0)
    real_allfake_text_y.append(1)
  else:
    real_allfake_text_clusters.append(1)
    real_allfake_text_y.append(1)



real_allfake_text_X_raw = Dictionary(real_allfake_text_X)
real_human_text_X_raw = Dictionary(real_human_text_X)

real_allfake_corpus = [real_allfake_text_X_raw.doc2bow(line) for line in real_allfake_text_X]
real_human_corpus = [real_human_text_X_raw.doc2bow(line) for line in real_human_text_X]

real_allfake_model = TfidfModel(real_allfake_corpus)
real_human_model = TfidfModel(real_human_corpus)

real_allfake_vector = real_allfake_model[real_allfake_corpus]
real_human_vector = real_human_model[real_human_corpus]

real_allfake_X = corpus2csc(real_allfake_vector).T
real_human_X = corpus2csc(real_human_vector).T



sklearn_pca = TruncatedSVD(n_components = 2)
Y_sklearn = sklearn_pca.fit_transform(real_allfake_X)
kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')
fitted = kmeans.fit(Y_sklearn)
prediction = kmeans.predict(Y_sklearn)

clustering_df = pandas.DataFrame(Y_sklearn)
clustering_df.columns = ['x', 'y']
clustering_df['prediction'] = prediction



#---------------------ML Model - Unsupervised -> Supervised---------------------#




#---------------------| Classification - Logistic Regression and SVM |---------------------#


X = real_human_X
y = real_human_text_y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

lr = LogisticRegression()
logit = lr.fit(X_train,y_train)
y_predict = logit.predict(X_test)

print("Logistic Regression - Accuracy:", logit.score(X, y)*100,"%")

#Generating AUC curve for the model
y_pred_proba = lr.predict_proba(X_test)[::,1]
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print("Logistic Regression - AUC:", auc)


X = real_human_X
y = real_human_text_y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svc = SVC(probability=True)
svm = svc.fit(X_train,y_train)
y_predict = svm.predict(X_test)

print("SVM - Accuracy:", svm.score(X, y)*100,"%")

#Generating AUC curve for the model
y_pred_proba = svc.predict_proba(X_test)[::,1]
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print("SVM - AUC:", auc)


#---------------------Classification - Logistic Regression and SVM---------------------#
