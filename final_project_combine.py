#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import re
import collections
import matplotlib.cm as cm
from matplotlib import rcParams
from textblob import TextBlob
import seaborn as sns
from pandas import *

from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


# # EDA

# In[2]:


Chipotle = read_csv('Chipotle.csv')


# In[3]:


#separate date and time
Chipotle['time']= Chipotle['date'].apply(lambda x: x.split()[1])
Chipotle['date']= Chipotle['date'].apply(lambda x: x.split()[0])
Chipotle['date'] = pd.to_datetime(Chipotle['date'])


# In[4]:


Chipotle.head()


# In[5]:


print("Number of Stores of Chipotle Mexican Grill: ",len(Chipotle["business_id"].value_counts()))
print("Number of Reviews of Chipotle Mexican Grill: ",len(Chipotle))


# In[6]:


fig= plt.figure(figsize = (12,4))
sns.countplot(Chipotle["business_id"],order = Chipotle["business_id"].value_counts().index)
plt.title('Number of Reviews for Each Store')
plt.xlabel("Stores")
plt.ylabel("Num of Reviews")
plt.xticks([])


# In[7]:


largest_store = Chipotle[Chipotle["business_id"] =="gOBxVkHpqtjRRxHBIrpnMA"]
year_month_large_store = list(largest_store['date'].dt.to_period('M'))
year_month_large_store.sort()
fig= plt.figure(figsize = (12,4))
sns.countplot(largest_store['date'].dt.to_period('M'), order = np.unique(year_month_large_store))
plt.title('Number of Review Change across time of the largest store')
plt.xticks(rotation = 90)


# In[8]:


month = list(pd.DatetimeIndex(Chipotle['date']).month)
month.sort()
fig= plt.figure(figsize = (12,4))
sns.countplot(pd.DatetimeIndex(Chipotle['date']).month, order = np.unique(month))
plt.title('Number of Review Change across time of all stores')
plt.xlabel('Month')
plt.xticks(rotation = 90)


# In[9]:


fig= plt.figure(figsize = (12,4))
sns.countplot(Chipotle['state'],order = Chipotle["state"].value_counts().index)
plt.title('Number of Review Change across states of all stores')
plt.xlabel('States')
plt.xticks(rotation = 90)


# In[10]:


main_cities = list(Chipotle["city"].value_counts().index[0:30])
Chipotle_main_cities = Chipotle[Chipotle['city'].isin(main_cities)]

fig= plt.figure(figsize = (10,4))
sns.countplot(Chipotle_main_cities['city'],order = Chipotle_main_cities["city"].value_counts().index, hue = Chipotle_main_cities['state'])
plt.title('Number of Review Change across main cities of partial stores')
plt.xlabel('city')
plt.xticks(rotation = 90)


# # Text Preprocessing

# In[11]:


Chipotle['text'][0]


# In[12]:


#decontraction
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"didn\'t", "did not", phrase)
    phrase = re.sub(r"don\'t", "do not", phrase)
    phrase = re.sub(r"o\'clock", "clock", phrase)
    phrase = re.sub(r"couldn\'t", "could not", phrase)
    phrase = re.sub(r"that\'s", "that is", phrase)       
    phrase = re.sub(r"go-around", "go around", phrase)  
    # general
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    #phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# In[13]:


# return to words
Chipotle['review'] = Chipotle['text'].apply(lambda x: decontracted(x))

# to lower
Chipotle['review'] = Chipotle['review'].apply(lambda x: x.lower())
# remove stop words
stop_words = stopwords.words('english')
Chipotle['review'] = Chipotle['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# remove non-word and punctuations
rm_punc = re.compile(r"\W")
shrink_spaces = re.compile(r"\s+")
Chipotle['review'] = Chipotle['review'].apply(lambda x: rm_punc.sub(' ', x))
Chipotle['review'] = Chipotle['review'].apply(lambda x: shrink_spaces.sub(' ', x))


# In[14]:


# # handle negation
# sub_nt = re.compile(r"n't")
# handle_not = re.compile(r"not (?=[a-zA-Z]+)")
# df['text'] = df['text'].apply(lambda x: sub_nt.sub(r' not', x))
# df['text'] = df['text'].apply(lambda x: handle_not.sub(r'not_', x))


# In[15]:


nltk.download('wordnet')
# import these modules 
import nltk
from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer() 
# lemmantize
Chipotle['review'] = Chipotle['review'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))


# In[16]:


Chipotle['review'].iloc[0]


# # Sentiment Analysis

# In[17]:


def sentiment(x):
    sentiment = TextBlob(x)
    return sentiment.sentiment.polarity


# In[18]:


Chipotle['sentiment_socre'] = Chipotle['review'].apply(sentiment)


# In[19]:


Chipotle.head()


# In[20]:


Chipotle['sentiment'] = ''
Chipotle['sentiment'][Chipotle['sentiment_socre'] > 0] = 'positive'
Chipotle['sentiment'][Chipotle['sentiment_socre'] <= 0] = 'negative'


# In[21]:


plt.figure(figsize=(6,3))
ax = sns.countplot(Chipotle['sentiment'])
plt.title('Review Sentiments')


# ### Collect Reviews and Count Words

# In[22]:


review_list_ch = Chipotle['review'].tolist()
review_all_ch = ' '.join(review_list_ch)
word_list_ch = [word for word in review_all_ch.lower().split() if word not in STOPWORDS]
counted_words = collections.Counter(word_list_ch)
words = []
counts = []
for letter, count in counted_words.most_common(50):
    words.append(letter)
    counts.append(count)


# ### Calculate Word Occurence Ratio in Positive / Negative Reviews

# In[23]:


pcount = []
ncount = []
for word in words:
    pcount.append(sum(Chipotle[Chipotle['sentiment'] == "positive"]["text"].str.contains(word)))
    ncount.append(sum(Chipotle[Chipotle['sentiment'] == "negative"]["text"].str.contains(word)))


# In[24]:


pcount_normalized = np.asarray(pcount)/Chipotle.loc[Chipotle['sentiment'] == "positive"].shape[0]
ncount_normalized = np.asarray(ncount)/Chipotle.loc[Chipotle['sentiment'] == "negative"].shape[0]


# In[25]:


word_pos_neg_cnt_df = pd.DataFrame({'positive ratio': pcount_normalized, 'negative ratio': ncount_normalized}, index=words)
word_pos_neg_cnt_df.head()


# # Time Series Sentiment Analysis

# ## Preliminary Analysis

# ### Take Stars vs Date

# In[26]:


df_ch_star = Chipotle[['stars_y', 'date']]
df_ch_star.head()


# ### Convert to Monthly Change

# In[27]:


ch_star_monthly_change = df_ch_star.set_index('date').resample('M').stars_y.mean()
ch_star_monthly_change = ch_star_monthly_change.dropna()
ch_star_monthly_change.head()
# df_ch.set_index('date').groupby('sentiment').get_group('positive').resample('M').count().sentiment


# ### Stars vs Month

# In[28]:


fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# ch_star_monthly_change = df_ch_star_vs_date.groupby(pd.Grouper(freq='30D', base=0, label='right')).mean()
ch_star_monthly_change.plot.line(label='Stars', ax=ax, rot=0, figsize=(10, 5), title="Chiptole Avg Stars over Time (avg per month)")


# ### Total positive / negative reviews per month

# In[29]:


# monthly total positive rows
Chipotle_pos_cnt = Chipotle.set_index('date').groupby('sentiment').get_group('positive').resample('M').count().sentiment
Chipotle_neg_cnt = Chipotle.set_index('date').groupby('sentiment').get_group('negative').resample('M').count().sentiment
print(Chipotle_pos_cnt.head())
print()
print(Chipotle_neg_cnt.head())


# In[30]:


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
Chipotle_pos_cnt.plot.line(label='Number of Positive Reviews', ax=ax, rot=0, figsize=(10, 5), title="Chipotle Number of Positive Reviews over Time (avg per month)", style='-')


# In[31]:


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
Chipotle_neg_cnt.plot.line(label='Number of Negative Reviews', ax=ax, rot=0, figsize=(10, 5), title="Chipotle Number of Negative Reviews over Time (avg per month)", style='-')


# ## Occurence of top frequent word in positive / negative reviews per month: step by step

# ### Pick a word as example

# In[32]:


# current woi (word of interest)
idx = 1
words[idx]


# ### All reviews containing keyword

# In[33]:


woi = words[1]
Chipotle_woi = Chipotle.loc[Chipotle['text'].str.contains(woi)]
Chipotle_woi.head()


# ### Positive / negative reviews containing keyword

# In[34]:


Chipotle_woi_pos_cnt = Chipotle_woi.set_index('date').groupby('sentiment').get_group('positive').resample('M').count().sentiment
Chipotle_woi_neg_cnt = Chipotle_woi.set_index('date').groupby('sentiment').get_group('negative').resample('M').count().sentiment
print(Chipotle_woi_pos_cnt.head())
print()
print(Chipotle_woi_neg_cnt.head())


# ### Occurence ratio in positive reviews and plot

# In[35]:


Chipotle_woi_pos_ratio = Chipotle_woi_pos_cnt.divide(Chipotle_pos_cnt, fill_value=0.0)
Chipotle_woi_pos_ratio.plot.line(rot=0, figsize=(10, 5), title="Monthly occurence of {} in positive reviews".format(woi))


# #### Occurence ratio in negative reviews and plot

# In[36]:


Chipotle_woi_neg_ratio = Chipotle_woi_neg_cnt.divide(Chipotle_neg_cnt, fill_value=0.0)
Chipotle_woi_neg_ratio.plot.line(rot=0, figsize=(10, 5), title="Monthly occurence of {} in negative reviews".format(woi))


# # Examine All Frequent Words

# In[37]:


def GetStarsInDateRange(x, start, end, colname):
    x_df = x.to_frame().reset_index('date')
    return x_df.loc[(x_df['date'] >= start) & (x_df['date'] <= end)].set_index('date')[colname]


# In[38]:


start_date = '2012-01-01'
end_date = '2018-12-31'

for woi in words:
    Chipotle_woi = Chipotle.loc[Chipotle['text'].str.contains(woi)]
    Chipotle_woi.head()
    Chipotle_woi_pos_cnt = Chipotle_woi.set_index('date').groupby('sentiment').get_group('positive').resample('M').count().sentiment
    Chipotle_woi_neg_cnt = Chipotle_woi.set_index('date').groupby('sentiment').get_group('negative').resample('M').count().sentiment

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    stars_plot = GetStarsInDateRange(ch_star_monthly_change, start_date, end_date, 'stars_y')
    stars_plot.plot.line(label='Stars', ax=ax, rot=0, figsize=(10, 5), title="Chipotle Avg Stars over Time (avg per month)", style='-')
    ax_r = ax.twinx()

    Chipotle_woi_pos_ratio = Chipotle_woi_pos_cnt.divide(Chipotle_pos_cnt, fill_value=0.0)
    Chipotle_woi_pos_ratio = Chipotle_woi_pos_ratio.dropna()
    woi_pos_ratio_plot = GetStarsInDateRange(Chipotle_woi_pos_ratio, start_date, end_date, 'sentiment')
    woi_pos_ratio_plot.plot.line(label='{} in positive'.format(woi), ax=ax_r, rot=0, style='g--')

    Chipotle_woi_neg_ratio = Chipotle_woi_neg_cnt.divide(Chipotle_neg_cnt, fill_value=0.0)
    Chipotle_woi_neg_ratio = Chipotle_woi_neg_ratio.dropna()
    woi_neg_ratio_plot = GetStarsInDateRange(Chipotle_woi_neg_ratio, start_date, end_date, 'sentiment')
    woi_neg_ratio_plot.plot.line(label='{} in negative'.format(woi), ax=ax_r, rot=0, style='r--')

    fig.legend()


# In[39]:


words


# In[ ]:


start_date = '2012-01-01'
end_date = '2018-12-31'

woi = "burrito"
Chipotle_woi = Chipotle.loc[Chipotle['text'].str.contains(woi)]
Chipotle_woi.head()
Chipotle_woi_pos_cnt = Chipotle_woi.set_index('date').groupby('sentiment').get_group('positive').resample('M').count().sentiment
Chipotle_woi_neg_cnt = Chipotle_woi.set_index('date').groupby('sentiment').get_group('negative').resample('M').count().sentiment

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
stars_plot = GetStarsInDateRange(ch_star_monthly_change, start_date, end_date, 'stars_y')
stars_plot.plot.line(label='Stars', ax=ax, rot=0, figsize=(10, 5), title="Chipotle Avg Stars over Time (avg per month)", style='-')
ax_r = ax.twinx()

Chipotle_woi_pos_ratio = Chipotle_woi_pos_cnt.divide(Chipotle_pos_cnt, fill_value=0.0)
Chipotle_woi_pos_ratio = Chipotle_woi_pos_ratio.dropna()
woi_pos_ratio_plot = GetStarsInDateRange(Chipotle_woi_pos_ratio, start_date, end_date, 'sentiment')
woi_pos_ratio_plot.plot.line(label='{} in positive'.format(woi), ax=ax_r, rot=0, style='g--')
fig.legend()


# # Topic Modeling

# In[ ]:


# tokenize each sentence into a list of words, removing punctuations
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(Chipotle['review']))


# In[ ]:


#Stemming
wnl = stem.WordNetLemmatizer()
data_words = [[wnl.lemmatize(word) for word in sentence] for sentence in data_words]


# In[ ]:


#create bag-of-words for each narrative
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=20) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=10)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
# print(trigram_mod[bigram_mod[data_words[0]]])
print(bigram_mod[data_words[0]])


# In[ ]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[ ]:


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])


# In[ ]:


# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
# print(corpus[:1])


# ## find the optimal number of topics for LDA

# In[ ]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word,random_seed=101)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

        
        

    return model_list, coherence_values


# In[ ]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, 
                                                 corpus=corpus, 
                                                 num_topics=num_topics, 
                                                 id2word=id2word,
                                                 random_seed=101)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        
       lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                            id2word=id2word,
#                                            num_topics=5, 
#                                            random_state=100,
#                                            update_every=1,
#                                            chunksize=100,
#                                            passes=10,
#                                            alpha='auto',
#                                            per_word_topics=True) 
        
        

    return model_list, coherence_values
                                                   
# Can take a long time to run.
mallet_path = 'mallet-2.0.8/bin/mallet' # update this path
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=10, step=1)


# In[ ]:


# Can take a long time to run.
mallet_path = 'mallet-2.0.8/bin/mallet' # update this path
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=10, step=1)


# In[ ]:


# Show graph
limit=10; start=2; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[ ]:


for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# ## Finding topics Using LDA model

# In[ ]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=6, 
                                           random_state=101,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.


# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# - Explanation for Perplexity:Perplexity is a measurement of how well a probability model predicts a test data.
# - Low perplexity is good and high perplexity is bad since the perplexity is the exponentiation of the entropy (a measure of uncertainty or randomness).

# In[ ]:


# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis


# In[ ]:


# pyLDAvis.save_html(vis, 'lda.html')


# ## Finding topics Using optimal LDA model

# In[ ]:


optimal_model = model_list[3]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


# In[ ]:


# # Compute Perplexity
# print('\nPerplexity: ', optimal_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=optimal_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[ ]:


# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(optimal_model, corpus, id2word)
vis


# # Try

# In[ ]:


num_of_topics= [1,2,3,4,5,6,7,8,9,10,11,12]

# Build LDA model
def find_n_of_topics(x):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=x, 
                                               random_state=101,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    return(CoherenceModel(model=lda_model, 
                          texts=data_lemmatized, 
                          dictionary=id2word, 
                          coherence='c_v').get_coherence())

coherence_num_of_topics = [num_of_topics.apply(lambda x: find_n_of_topics(x))]


# In[ ]:


num_of_topics= [1,2,3,4,5,6,7,8,9,10,11,12]
coherence_num_of_topics = list()
# Build LDA model
for n in num_of_topics:
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics= n, 
                                               random_state=101,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    coherence_num_of_topics.append(CoherenceModel(model=lda_model, 
                          texts=data_lemmatized, 
                          dictionary=id2word, 
                          coherence='c_v').get_coherence())


# In[ ]:


# Show graph
plt.plot(num_of_topics, coherence_num_of_topics)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[ ]:


for m, cv in zip(num_of_topics, coherence_num_of_topics):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# In[ ]:


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics= 3, 
                                           random_state=101,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[ ]:


# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.


# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[ ]:


# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis


# # Action Items:
# - 调参，增加topic个数
# - 删去不重要的词
# - 好的topic 不好的topic based on sentiment analysis

# In[ ]:




