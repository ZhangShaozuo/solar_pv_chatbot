from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from rank_bm25 import BM25Okapi
# Function to preprocess text (simple version for demonstration)
def preprocess_text(text):
    return text.lower().split()

# Function to vectorize a list of words using the Word2Vec model
def vectorize(words, model):
    vector = np.zeros(model.vector_size)
    count = 0
    for word in words:
        if word in model.wv:
            vector += model.wv[word]
            count += 1
    if count > 0:
        vector /= count
    return vector

def find_most_relevant_paragraph(paragraphs, query):
    # Preprocess paragraphs and query
    tokenized_paragraphs = [preprocess_text(paragraph) for paragraph in paragraphs]
    tokenized_query = preprocess_text(query)

    # Train Word2Vec model
    model = Word2Vec(sentences=tokenized_paragraphs, vector_size=100, window=5, min_count=1, workers=4)

    # Vectorize query and paragraphs
    query_vector = vectorize(tokenized_query, model)
    paragraph_vectors = [vectorize(paragraph, model) for paragraph in tokenized_paragraphs]

    # Compute similarity between query and each paragraph
    similarities = [cosine_similarity([query_vector], [paragraph_vector])[0][0] for paragraph_vector in paragraph_vectors]

    # Identify the index of the most relevant paragraph
    # Identify the top 3 most relevant paragraphs
    most_relevant_paragraph_indices = np.argsort(similarities)[-5:]
    most_relevant_paragraph_indices = np.flip(most_relevant_paragraph_indices)

    # most_relevant_paragraph_index = similarities.index(max(similarities))
    # Return the most relevant paragraph
    op = []
    for i in most_relevant_paragraph_indices:
        op.append(paragraphs[i])
    return op

def rank_sentence_transformer(paragraphs, query):
    model = SentenceTransformer('paraphrase-albert-small-v2')
    query_embedding = model.encode(query.lower().strip(), convert_to_tensor=True)
    sentence_embeddings = model.encode(paragraphs, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
    ## retrieve the top 3 indexes with highest cosine similarity scores
    try: 
        top_results = torch.topk(cos_scores, k=2).indices
    except:
        top_results = torch.topk(cos_scores, k=1).indices
    op = []
    for i in top_results:
        op.append(paragraphs[i])
    return op

def rank_bm25(paragraphs, query):
    bm25 = BM25Okapi(paragraphs)
    op = bm25.get_top_n(query, paragraphs, n=2)
    return op


