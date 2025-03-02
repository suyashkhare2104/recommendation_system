from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm

def vectorize_articles(csv_path: str, output_dir: str, sample_size: int = 5000):
    """
    Load articles from CSV, sample a subset, and create vectors
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess articles
    print("Loading articles from CSV...")
    df = pd.read_csv(csv_path, sep=';')
    df['published_date'] = pd.to_datetime(df['published_date']).dt.date
    df = df.dropna(subset=['title'])
    
    # Sample articles
    print(f"Sampling {sample_size} articles from {len(df)} total articles...")
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)  # for reproducibility
    
    # Convert DataFrame to list of dictionaries
    articles = df.to_dict('records')
    
    # Create vectors
    print("Creating vectors...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [article['title'] for article in articles]
    vectors = model.encode(texts, show_progress_bar=True)
    
    # Save articles and vectors
    print("Saving files...")
    with open(os.path.join(output_dir, 'articles.pkl'), 'wb') as f:
        pickle.dump(articles, f)
    
    with open(os.path.join(output_dir, 'vectors.npy'), 'wb') as f:
        np.save(f, vectors)
    
    # Save metadata about the sample
    metadata = {
        'total_articles': len(df),
        'sample_size': sample_size,
        'vector_dim': vectors.shape[1],
        'topics': df['topic'].value_counts().to_dict()
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Processed {len(articles)} articles")
    print("\nTopic distribution in sample:")
    for topic, count in df['topic'].value_counts().items():
        print(f"{topic}: {count}")
    print(f"\nFiles saved to {output_dir}")

if __name__ == "__main__":
    csv_path = "data/news.csv"
    output_dir = "data/processed"
    vectorize_articles(csv_path, output_dir, sample_size=5000) 