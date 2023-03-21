#!/usr/bin/env python
# coding: utf-8

# # Topic Modeling
# ## Data Exploration
# ## Topic Modeling
# ## Human in the Loop Input
# 
# 
# 

# Links:
# - https://data.stackexchange.com/
# - https://data.stackexchange.com/health/query/new
# 
# 
# 
# 

# In[ ]:


'''
-- RUN  HERE: https://data.stackexchange.com/money
-- GOAL: Finance Posts (Most Popular Questions) 


SELECT TOP 10000 

    Posts.Id AS POST_ID
	,Posts.CreationDate
	,Posts.OwnerUserId
    ,Posts.Title AS SUBJ
	,Posts.Body AS BODY
	,Posts.PostTypeId
    ,Posts.ViewCount

FROM Posts
INNER JOIN PostTags ON Posts.Id = PostTags.PostId


WHERE 
Posts.PostTypeId = 1

GROUP BY 

--Tags.TagName
--	,Tags.id
--	,
    Posts.Id
	,Posts.CreationDate
	,Posts.OwnerUserId
	,Posts.Body
	,Posts.ViewCount
	,Posts.Title
	,Posts.PostTypeId 
    --PT.Name
    

ORDER BY Posts.ViewCount DESC# Calculations and Graphing
'''


# In[2]:



import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sb
from pandas_profiling import ProfileReport as PR 
#import itertools
#import matplotlib.ticker as ticker

#ML
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#Multiple Outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# In[3]:


df=pd.read_csv(r'QueryResults_Finance_Mar21.csv')

df.head(2)
df.columns


# # Topic Modeling 
# - Meaningful words are Binary Vectors (checklists whether the words appear or not) 
# - Frequently associated words make "topics" 
# - Email Titles can be helpful. Body text requires more complexity. Ushur can do! :-) 
# 
# 
# 

# In[20]:


df.head(20)


# In[28]:





# In[29]:


pd.set_option('display.max_colwidth', None)


# In[37]:


pd.DataFrame(df.loc[12:15])


# In[4]:


df.columns


# In[5]:


# Count Vectorization for Body Text
count_vect_body = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
doc_term_matrix_body = count_vect_body.fit_transform(df['BODY'].values.astype('U'))


# In[6]:


# Count Vectorization for Title Text
count_vect_title = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
doc_term_matrix_title = count_vect_title.fit_transform(df['SUBJ'].values.astype('U'))


# In[7]:


#Title Vectorization Sparse Matrix
pd.DataFrame.sparse.from_spmatrix(doc_term_matrix_title, columns=count_vect_title.get_feature_names() ).head()


# In[8]:


title_words=pd.DataFrame.sparse.from_spmatrix(doc_term_matrix_title, columns=count_vect_title.get_feature_names() )
title_words.head()
body_words=pd.DataFrame.sparse.from_spmatrix(doc_term_matrix_body, columns=count_vect_body.get_feature_names() )
body_words.head()


# # Keyword Metrics

# In[9]:


sb.set()
top_10_title=pd.Series(title_words.sum(), name='Title').sort_values(ascending=False)[:20]
top_10_body=pd.Series(body_words.sum(), name='Body').sort_values(ascending=False)[:20]


# In[11]:


#Body Keywords
top_10_body.plot(kind='barh',  title='Frequent Body Mentions')


# In[12]:


top_10_title.plot(kind='barh', title='Frequent Title Mentions')


# # Title Text Topic Modeling

# In[13]:


# Pick Six Topics- they'll bring up certain words that co-occur. 
LDA_title = LatentDirichletAllocation(n_components=6, random_state=42)
LDA_title.fit(doc_term_matrix_title)
topic0_list=LDA_title.components_[0].argsort()[-10:]
topic0_list

# List Comprehension to get keywords
[count_vect_title.get_feature_names()[ind] for ind in topic0_list]


# In[14]:


#Title 6 

LDA_title = LatentDirichletAllocation(n_components=6, random_state=42)
LDA_title.fit(doc_term_matrix_title)
topic0_list=LDA_title.components_[0].argsort()[-10:]
topic0_list
df_LDA_title=pd.DataFrame()
df_LDA_title_weight=pd.DataFrame()


for ind, topic in enumerate(LDA_title.components_):
    df_LDA_title['Topic_'+str(ind)]=[count_vect_title.get_feature_names()[i] for i in topic.argsort()[-10:]]
    df_LDA_title_weight['Weight_'+str(ind)]=[topic[i] for i in topic.argsort()[-10:]]
df_LDA_title
df_LDA_title_weight
df_LDA_title.to_html('Title_Topics_6.html')


# In[15]:



plt.rcParams["figure.figsize"] = (20,15)
plt.subplots_adjust(hspace=2.5)
fig, axs = plt.subplots(nrows=2, ncols=3)
for ind, col in enumerate(df_LDA_title): 
    col1=df_LDA_title[col]
    col2=df_LDA_title_weight['Weight_'+str(ind)]
#    col1
    data=pd.DataFrame(pd.concat([col1,col2], axis=1))
#        data=[[col1.values,col2.]], columns=[col1.name, col2.name])
    
    ax = plt.subplot(2, 3, ind + 1)
    data.plot(kind='barh', x=col, y='Weight_'+str(ind), ax=ax)
    #ax.set_title(ticker.upper())
    ax.get_legend().remove()
    ax.set_xlabel("")
    plt.yticks(fontsize=16) 


# In[19]:


fig


# # Explore what the rankings tell the user

# In[17]:


# Example w/ Ten
LDA_title = LatentDirichletAllocation(n_components=10, random_state=42)
LDA_title.fit(doc_term_matrix_title)
topic0_list=LDA_title.components_[0].argsort()[-10:]
topic0_list
df_LDA_title=pd.DataFrame()

df_LDA_title
for ind, topic in enumerate(LDA_title.components_):
    df_LDA_title[ind]=[count_vect_title.get_feature_names()[i] for i in topic.argsort()[-10:]]
df_LDA_title
df_LDA_title.to_html('Title_Topics_10.html')


# In[18]:


#topic_values = LDA_title.transform(doc_term_matrix_title)
#topic_values.shape
pd.DataFrame(topic_values)
#.plot(kind='heatmap')
#.plot()


# # Human in the Loop
#  - A person needs to look over the keywords generated and determine WHAT the topics are
#  - Choosing the number of topics is a user decision
#  - E.g. topic 8 above is about daily hygiene while 9 is about an operation
#  - Some topics may require additional human parsing 
#  - Contrast this to the same process in the body
#  - Some emails may contain multiple topics 
