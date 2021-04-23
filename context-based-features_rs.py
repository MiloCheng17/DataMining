from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from ast import literal_eval


# Get citation data
fileobj = 'savedrecs-440.xls'
metadata = pd.read_excel(fileobj) 
indices = pd.Series(metadata.index, index=metadata['Article Title']).drop_duplicates()

def get_recommendations(title, cosine_sim):
    # Get the index of the article that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all titles with that article
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the articles based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar articles
    sim_scores = sim_scores[0:11]

    # Get the article indices
    article_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['Article Title'].iloc[article_indices]

### Get corresponding author ###
def get_last_author(x):
    authors = x.split(';')
    return authors[-1]

### Get Keywords ###
def get_keywords_list(x):
    if isinstance(x, str):
        names = [i for i in x.split(';')]
            #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names
    return []

### Clean data ###
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if author exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

### for selected features ###
#features = ['Article Title', 'Authors', 'Author Keywords', 'Keywords Plus', 'Research Areas'] #'WoS Categories', '180 Day Usage Count', 'Times Cited, WoS Core' 
features = ['Author Keywords', 'Author Full Names', 'Research Areas']

#for feature in features:
#    metadata[feature] = metadata[feature].apply(get_keywords_list)
#metadata['Author'] = metadata['Authors'].apply(get_last_author)

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)
#print(metadata[features].head(3))

def create_soup(x):
    return ' '.join(x['Author Keywords']) + ' ' + ' '.join(x['Article Title']) + ' ' + x['Authors'] + ' ' + ' '.join(x['Research Areas'])

## Create a new soup feature
metadata['soup'] = metadata.apply(create_soup, axis=1)
#print(metadata['soup'].head(2))

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])
#print(count_matrix.shape)

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
# Reset index of the main DataFrame and construct reverse mapping
metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['Article Title'])

#print(get_recommendations('Enabling microbial syringol conversion through structure-guided protein engineering', cosine_sim2))
print(get_recommendations('A promiscuous cytochrome P450 aromatic O-demethylase for lignin bioconversion', cosine_sim2))
