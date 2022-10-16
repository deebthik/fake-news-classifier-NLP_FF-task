
import os
import pandas as pd

# importing word_tokenize to use in function below
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')


cwd = os.path.abspath(os.getcwd())

real_df = pd.read_csv(cwd + "/True.csv")
fake_df = pd.read_csv(cwd + "/Fake.csv")

print("Number of Real articles: " + str(len(real_df)))

print("Number of Fake articles: " + str(len(fake_df)))

# Viewing dataframe
real_df.head()


# Dropping title, subject and data columns since we cant use these in AI-text
real_df.drop(labels=['title', 'subject', 'date'], axis=1, inplace=True)
fake_df.drop(labels=['title', 'subject', 'date'], axis=1, inplace=True)

real_df.head()


# Real articles that contain "Reuters" in them
real_df[real_df['text'].str.contains("Reuters")]

# Fake articles that contain "Reuters" in them
fake_df[fake_df['text'].str.contains("Reuters")]

# Remove 'Reuters) - ' and anything that comes before it
real_df['text'] = real_df['text'].apply(lambda x: x.split('Reuters) - ')[-1])

# Remove 'Reuters' even if it doesnt follow format above
real_df['text'] = real_df['text'].apply(lambda x: x.replace('Reuters', ''))


# Print remaining rows containing Reuters - expecting empty DataFrame
print(real_df[real_df['text'].str.contains("Reuters")])

# Repeat for Fake df
fake_df['text'] = fake_df['text'].apply(lambda x: x.split('Reuters) - ')[-1])
fake_df['text'] = fake_df['text'].apply(lambda x: x.replace('Reuters', ''))

# Print remaining rows containing Reuters - expecting empty DataFrame
print(fake_df[fake_df['text'].str.contains("Reuters")])


# Examples of text we want to clean

#Contains unicode space '\xa0'
real_df.text.iloc[13]

# Contains https link
fake_df.text.iloc[3]

import re

def clean_text(text):
  """Cleans newline characters, unix whitespace,
  and http links from text"""

  text = re.sub(r'\n', ' ', text)
  text = re.sub('\xa0', ' ', text)
  cleaned_text = re.sub('(http)(.+?)(?:\s)', ' ', text)   # 'http' followed by text up until space char

  return cleaned_text


 # Clean the text on all dfs

df_list = [real_df, fake_df]

for df in df_list:
  df['text'] = df['text'].apply(lambda x: clean_text(x))



real_df["group"] = "real"
fake_df["group"] = "fake_hum"

# Combining dfs
articles_df = real_df.append([fake_df], verify_integrity=True, ignore_index=True)

articles_df


# Are there duplicates?

articles_df[articles_df["group"] == 'fake_hum'] # of fake_hum articles: 21226
len(articles_df[articles_df["group"] == 'fake_hum'].text.unique()) # 16385 unique fake_hum articles


articles_df[articles_df["group"] == 'real'] # of real articles: 20253
len(articles_df[articles_df["group"] == 'real'].text.unique()) # 20043 unique real articles



# Yes, all 3 groups have some duplicates, particularly the fake_hum group with 4841 duplicate articles.
# fake_ai and real each have < 250 duplicates
# 5,120 duplicate articles total


# Dropping rows with duplicate articles
articles_df.drop_duplicates(subset=['text'], inplace=True, ignore_index=True)

articles_df



# Create column for article length

articles_df['length'] = articles_df['text'].apply(lambda x: len(x))


# Viewing the number of articles for each group

print("# of articles for real: " + str(len(articles_df[articles_df['group'] == 'real'])))
print("# of articles for fake_hum: " + str(len(articles_df[articles_df['group'] == 'fake_hum'])))



# Drop short articles
articles_df.drop(articles_df[articles_df['length'] < 400].index, inplace=True)

# Reset index
articles_df.reset_index(drop=True, inplace=True)

# New number of articles
print("# of articles for real: " + str(len(articles_df[articles_df['group'] == 'real'])))
print("# of articles for fake_hum: " + str(len(articles_df[articles_df['group'] == 'fake_hum'])))




import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(x='group', y='length', data=articles_df)
'''
plt.xlabel("Group", fontsize= 12)
plt.ylabel("Length", fontsize= 12)
plt.title("Article Length by Group", fontsize=15)'''




# Longest fake_ai article
articles_df[articles_df.group == 'fake_ai'].length.max()

# All articles that are too long
articles_df[articles_df.length > 5711]



def article_crop(article, char_limit):
  """Given a DataFrame row that contains a 'text' column and a character limit, this crops
      the text to end at the sentence after the character limit is reached

  Params
  - row: a row of a Dataframe that contains 'text' column
  - char_limit: int, once text reaches this limit, the current sentence will become the final sentence in the text
  """


  # If text is longer than char limit
  if len(article) > char_limit:

    # Split text into two groups, before and after the character limit
    before_limit = article[: char_limit]
    after_limit = article[char_limit : ]

    # If after_limit contains more than just whitespace (check is needed b/c calling sent_tokenize()[0] on whitespace will fail)
    if len(nltk.sent_tokenize(after_limit)) > 0:

      sent_after_limit = nltk.sent_tokenize(after_limit)[0] # Take 1st sentence after char_limit

      new_text = before_limit + sent_after_limit
      return new_text

    # In this case, chars after char_limit only contain whitespace
    else:
      return before_limit

  else:
    return article


# Shows where data will be cut off at the limit
articles_df.text.iloc[63][5400:5711]

# Shows the final sentence being completed using article_crop()
article_crop(articles_df.text.iloc[63], 5711)[5400:]

# For texts longer than 5711 chars (the longest AI text), crop when the sentence at 5711 chars ends
articles_df['text'] = articles_df.text.apply(lambda x: article_crop(x, 5711))
articles_df

# Recomputing article lengths after crops
articles_df['length'] = articles_df['text'].apply(lambda x: len(x))


# Viewing new article lengths after crops
sns.boxplot(x='group', y='length', data=articles_df)
'''
plt.xlabel("Group", fontsize= 12)
plt.ylabel("Length", fontsize= 12)
plt.title("Article Length by Group", fontsize=15)'''

# articles that are still really long
articles_df[articles_df.length > 8000]

# Last part of articles that are still really long
# Both are sentences separated by colons and semicolons
articles_df.text.iloc[22681][-2500:]
articles_df.text.iloc[35379][-2500:]



# Tokenize sentences
articles_df['sentences'] = articles_df['text'].apply(lambda x: nltk.sent_tokenize(x))

articles_df


def avg_words_per_sent(text):
  """Feed a list of sentences (param:text) and
  returns the average number of words per sentence"""

  num_words_list = [] # List of word count per sentence

  for sent in range(len(text)):

    # Number of words in a sentence, excludes punctuation
    num_words = len([word for word in word_tokenize(text[sent]) if word.isalnum()])

    # Add number of words for sentence to the list
    num_words_list.append(num_words)


  # Compute the average
  avg_word_per_sentences = sum(num_words_list) / len(num_words_list)

  return avg_word_per_sentences

  # Compute avg words per sentence for each column
articles_df['words_per_sent'] = articles_df['sentences'].apply(lambda x: avg_words_per_sent(x))

articles_df.head()




sns.boxplot(x='group', y='words_per_sent', data=articles_df)
'''
plt.xlabel("Group", fontsize= 14)
plt.ylabel("Words per Sentence", fontsize= 14)
plt.title("Avg words per sentence by group", fontsize=18)'''


# Dropping articles with 100+ average words per sentence
articles_df.drop(articles_df[articles_df['words_per_sent'] > 100].index, inplace=True)

# Reset index
articles_df.reset_index(drop=True, inplace=True)

print("New count of articles")
print("# of articles for real: " + str(len(articles_df[articles_df['group'] == 'real'])))
print("# of articles for fake_hum: " + str(len(articles_df[articles_df['group'] == 'fake_hum'])))


# Replotting avg words per sentence
sns.boxplot(x='group', y='words_per_sent', data=articles_df)
'''
plt.xlabel("Group", fontsize= 14)
plt.ylabel("Words per Sentence", fontsize= 14)
plt.title("Avg words per sentence by group", fontsize=16)'''


# Plotting probability distribution due to uneven # of samples between groups
sns.displot(articles_df, x='words_per_sent', hue='group', stat='probability', common_norm=False)
'''
plt.ylabel("Probability", fontsize= 14)
plt.xlabel("Avg. words/sentence", fontsize= 14)
plt.title("Distribution of avg. words/sentence by group", fontsize=16)'''


# Create column of lowercase text
articles_df['text_lower'] = articles_df['text'].apply(lambda x: x.lower())

articles_df.head()


from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

stopwords = set(stopwords.words('english'))

def tokenize_text(article):
  """ take a text string and
  - tokenizes words
  - lowers case
  - removes stopwords
  - remove non alpha tokens

  :param article_list: list of strings
  :return: list of cleaned text
  """

  tok_text = nltk.word_tokenize(article)
  tok_text = [word.lower() for word in tok_text]
  tok_text = [word for word in tok_text if word not in stopwords]
  tok_text = [word for word in tok_text if word.isalpha()]


  return tok_text


# Create a list of text articles for each group
real_text_list = list(articles_df.text[articles_df.group == "real"])
fakehum_text_list= list(articles_df.text[articles_df.group == "fake_hum"])


# Tokenize and clean the lists from stopwords
# each result is list of lists containing words for each article
real_word_list = [tokenize_text(article) for article in real_text_list]
fakehum_word_list = [tokenize_text(article) for article in fakehum_text_list]


#flatten the list of lists into a single list of words
flat_real_word_list = [word for article in real_word_list for word in article]
flat_fakehum_word_list = [word for article in fakehum_word_list for word in article]


from collections import Counter

# 50 most common words for real
real_cnt = Counter(flat_real_word_list)
real_most_common = real_cnt.most_common(50)

# Show top 30
real_most_common[:30]


# 50 most common words for fake_hum
fakehum_cnt = Counter(flat_fakehum_word_list)
fakehum_most_common = fakehum_cnt.most_common(50)

# Show top 30
fakehum_most_common[:30]







real_words = []
for i in range(len(real_most_common)):
  # Add the word, not the frequency count
  real_words.append(real_most_common[i][0])

# List of most 50 common words in real
real_words



fake_words = []
for i in range(len(fakehum_most_common)):
  # Add the word, not the frequency count
  fake_words.append(fakehum_most_common[i][0])

# List of most 50 common words in fake
fake_words


# Create set from the 50 most common words from both human-authored groups
human_words = set(real_words + fake_words)

print(len(human_words))
print("")
print(human_words)

# Tag most common human-authored words with parts of speech
nltk.pos_tag(human_words)


# keep only nouns in human_words
tagged_human_words = nltk.pos_tag(human_words)

human_nouns = [word_tag_pair[0] for word_tag_pair in tagged_human_words if word_tag_pair[1] == 'NN']
human_nouns


# Remove the following words: way, image, time, year, video, get, week
human_nouns = [word for word in human_nouns if word not in ['way', 'time', 'image', 'year', 'video', 'get', 'week']]

# Final list
human_nouns


def contains_word_in_list(text, word_list):
  """ Given a text string, returns true if string contains any word in word_list
  else returns false

  :param - text -- string
  : word_list - a list of words
  """
  if any(word in text for word in word_list):
    return True
  else:
    return False


# Create column
articles_df["contains_human_word"] = articles_df['text_lower'].apply(lambda x: contains_word_in_list(x, human_nouns))

# Lets recount how many articles we have before dropping more AI articles

print("# of articles for real: " + str(len(articles_df[articles_df['group'] == 'real'])))
print("# of articles for fake_hum: " + str(len(articles_df[articles_df['group'] == 'fake_hum'])))


# Drop AI_articles without human news words
articles_df.drop(articles_df[(articles_df.group=="fake_ai") & (articles_df.contains_human_word==False)].index, inplace=True)

# Drop contains_human_word column because no longer needed
articles_df.drop(['contains_human_word'], axis=1, inplace=True)

# Reset index
articles_df.reset_index(drop=True, inplace=True)

print("# of articles for real: " + str(len(articles_df[articles_df['group'] == 'real'])))
print("# of articles for fake_hum: " + str(len(articles_df[articles_df['group'] == 'fake_hum'])))








from afinn import Afinn


afinn = Afinn(language='en')

# Example of sentiment scores
print('Score for "I hate this and it sucks": ', afinn.score("I hate this and it sucks"))
print('Score for "I love it and it\'s great!": ', afinn.score("I love it and it's great!"))

# Create sentiment column
articles_df['sentiment'] = articles_df.text.apply(lambda x: afinn.score(x))

'''
sns.boxplot(x='group', y='sentiment', data=articles_df)

plt.xlabel("Group", fontsize= 14)
plt.ylabel("Sentiment", fontsize= 14)
plt.title("Sentiment by group", fontsize=18)'''

# Final df
articles_df.head()



# articles_df.describe()
print("Avg length - real: " + str(articles_df[articles_df['group']== 'real'].length.mean()))
print("Avg length - fake_hum: " + str(articles_df[articles_df['group']== 'fake_hum'].length.mean()))







from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

real_text = articles_df[articles_df['group'] == "real"][['text','group']][:1000]
fake_text = articles_df[articles_df['group']=='fake_hum'][['text','group']][:500]

clustering_text_X = real_text['text'].append(fake_text['text'])


# Classification - Real vs. All Fake
# X = real_allfake_text_X | y = real_allfake_text_y
real_allfake_text_X = real_text['text'].append(fake_text['text'])
real_allfake_text_y_raw = real_text['group'].append(fake_text['group'])
real_allfake_text_y = []
real_allfake_text_clusters = []
for val in real_allfake_text_y_raw:
  if val == 'real':
    real_allfake_text_clusters.append(1)
    real_allfake_text_y.append(0)
  elif val == 'fake_hum':
    real_allfake_text_clusters.append(2)
    real_allfake_text_y.append(1)
  else:
    real_allfake_text_clusters.append(0)
    real_allfake_text_y.append(1)



tf_idf_vectorizor3 = TfidfVectorizer()
tf_idf3 = tf_idf_vectorizor3.fit_transform(real_allfake_text_X)
tf_idf_norm3 = normalize(tf_idf3)
X_real_allfake = tf_idf_norm3.toarray()


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
sklearn_pca = PCA(n_components = 2)
Y_sklearn = sklearn_pca.fit_transform(X_real_allfake)
kmeans = KMeans(n_clusters=3, max_iter=600, algorithm = 'auto')
fitted = kmeans.fit(Y_sklearn)
prediction = kmeans.predict(Y_sklearn)

clustering_df = pd.DataFrame(Y_sklearn)
clustering_df.columns = ['x', 'y']
clustering_df['prediction'] = prediction

plt.scatter(Y_sklearn[:,0],Y_sklearn[:,1], c = prediction);
plt.title("Clustering Algorithm on Principal Components of Data (Subset)", size = 18)
plt.gcf().set_size_inches(15,5)

plt.show()


plt.figure()
plt.scatter(Y_sklearn[:,0],Y_sklearn[:,1], c = real_allfake_text_clusters);
plt.title("Actual Classes on Principal Components of Data (Subset)", size = 18)
plt.gcf().set_size_inches(15,5)







from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

# Limiting to 16,000 to control for different samples sizes between groups
real_text = articles_df[articles_df['group'] == "real"][['sentences','group']][:16000]
fake_text = articles_df[articles_df['group']=='fake_hum'][['sentences','group']][:16000]

# Clustering
clustering_text_X = real_text['sentences'].append(fake_text['sentences'])


# Classification - Real vs. Human
# X = real_human_text_X | y = real_human_text_y
real_human_text_X = real_text['text'][:500].append(fake_text['text'])
real_human_text_y_raw = real_text['group'][:500].append(fake_text['group'])
real_human_text_y = []
for val in real_human_text_y_raw:
  if val == 'real':
    real_human_text_y.append(0)
  else:
    real_human_text_y.append(1)
    

# Classification - Real vs. All Fake
# X = real_allfake_text_X | y = real_allfake_text_y
real_allfake_text_X = real_text['sentences'].append(fake_text['sentences'][:8000])
real_allfake_text_y_raw = real_text['group'].append(fake_text['group'][:8000])
real_allfake_text_y = []
real_allfake_text_clusters = []
for val in real_allfake_text_y_raw:
  if val == 'real':
    real_allfake_text_clusters.append(2)
    real_allfake_text_y.append(0)
  elif val == 'fake_hum':
    real_allfake_text_clusters.append(0)
    real_allfake_text_y.append(1)
  else:
    real_allfake_text_clusters.append(1)
    real_allfake_text_y.append(1)







from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc

real_allfake_text_X_raw = Dictionary(real_allfake_text_X)
real_human_text_X_raw = Dictionary(real_human_text_X)

real_ai_corpus = [real_ai_text_X_raw.doc2bow(line) for line in real_ai_text_X]
real_allfake_corpus = [real_allfake_text_X_raw.doc2bow(line) for line in real_allfake_text_X]
real_human_corpus = [real_human_text_X_raw.doc2bow(line) for line in real_human_text_X]

real_allfake_model = TfidfModel(real_allfake_corpus)
real_human_model = TfidfModel(real_human_corpus)

real_allfake_vector = real_allfake_model[real_allfake_corpus]
real_human_vector = real_human_model[real_human_corpus]

real_allfake_X = corpus2csc(real_allfake_vector).T
real_human_X = corpus2csc(real_human_vector).T







from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

sklearn_pca = TruncatedSVD(n_components = 2)
Y_sklearn = sklearn_pca.fit_transform(real_allfake_X)
kmeans = KMeans(n_clusters=3, max_iter=600, algorithm = 'auto')
fitted = kmeans.fit(Y_sklearn)
prediction = kmeans.predict(Y_sklearn)

clustering_df = pd.DataFrame(Y_sklearn)
clustering_df.columns = ['x', 'y']
clustering_df['prediction'] = prediction


plt.scatter(Y_sklearn[:,0],Y_sklearn[:,1], c = prediction);
plt.title("Clustering Algorithm on Principal Components of Data (All Data)", size = 18)
plt.gcf().set_size_inches(15,5)

plt.figure()
plt.scatter(Y_sklearn[:,0],Y_sklearn[:,1], c = real_allfake_text_clusters);
plt.title("Actual Classes on Principal Components of Data (All Data)", size = 18)
plt.gcf().set_size_inches(15,5)

plt.show()




from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = real_allfake_X
y = real_allfake_text_y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
logit = lr.fit(X_train,y_train)
y_predict = logit.predict(X_test)

print("Accuracy: ", logit.score(X, y)*100,"%")



from sklearn.svm import SVC

X = real_allfake_X
y = real_allfake_text_y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svc = SVC()
svm = svc.fit(X_train,y_train)
y_predict = svm.predict(X_test)

print("Accuracy: ", svm.score(X, y)*100,"%")
