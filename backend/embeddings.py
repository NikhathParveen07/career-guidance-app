# ============================================
# backend/embeddings.py
# Encode queries and search Pinecone vector index
# ============================================


def encode_query(query_text, sentence_model):
    """
    Convert a student's interest text into a 384-dimensional embedding.
    The same model used to embed careers is used here,
    ensuring queries and career vectors exist in the same space.
    """
    return sentence_model.encode(query_text).tolist()


def search_careers(query_text, sentence_model, index, top_k=20):
    """
    Semantic search: encode the query and retrieve the top_k
    most similar careers from the Pinecone vector index.

    Returns a list of matches with metadata and cosine similarity scores.
    """
    query_vector = encode_query(query_text, sentence_model)

    results = index.query(
        vector         = query_vector,
        top_k          = top_k,
        include_metadata = True
    )

    return results['matches']
