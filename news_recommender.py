from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import pickle
import os

class NewsRecommender:
    def __init__(self, processed_dir: str):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_processed_data(processed_dir)

    def load_processed_data(self, processed_dir: str):
        """Load pre-computed articles and vectors"""
        articles_path = os.path.join(processed_dir, 'articles.pkl')
        vectors_path = os.path.join(processed_dir, 'vectors.npy')
        
        with open(articles_path, 'rb') as f:
            self.articles = pickle.load(f)
        
        with open(vectors_path, 'rb') as f:
            self.vectors = np.load(f)

    def get_recommendations(self, query: str, top_k: int = 5) -> List[Dict]:
        """Get top-k similar articles for a query"""
        if not self.articles:
            return []
        
        # Encode query
        query_vector = self.model.encode(query)
        
        # Calculate similarities
        similarities = np.dot(self.vectors, query_vector)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return top articles
        return [self.articles[idx] for idx in top_indices]

    def get_similar_articles(self, article_index: int, top_k: int = 5) -> List[Dict]:
        """Get top-k similar articles for a given article index"""
        if not self.articles or article_index >= len(self.articles):
            return []
        
        # Get the vector for the selected article
        article_vector = self.vectors[article_index]
        
        # Calculate similarities
        similarities = np.dot(self.vectors, article_vector)
        
        # Get top indices (excluding the article itself)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]  # Skip first (self)
        
        # Return top articles with similarity scores
        return [
            {
                'article': self.articles[idx],
                'similarity': float(similarities[idx])  # Convert to float for JSON serialization
            }
            for idx in top_indices
        ]

    def get_article_index(self, title: str) -> int:
        """Find article index by title"""
        for idx, article in enumerate(self.articles):
            if article['title'] == title:
                return idx
        return -1 