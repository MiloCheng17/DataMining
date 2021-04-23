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

tfidf = TfidfVectorizer(stop_words='english')
metadata['Abstract'] = metadata['Abstract'].fillna('')
tfidf_matrix = tfidf.fit_transform(metadata['Abstract'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and article titles
indices = pd.Series(metadata.index, index=metadata['Article Title']).drop_duplicates()
#print(indices[:10])

# Function that takes in article abstract as input and outputs most similar articles
def get_recommendations(title, cosine_sim=cosine_sim):
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
#print(get_recommendations('Enabling microbial syringol conversion through structure-guided protein engineering'))
print(get_recommendations('A promiscuous cytochrome P450 aromatic O-demethylase for lignin bioconversion'))

