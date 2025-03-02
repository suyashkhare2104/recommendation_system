import streamlit as st
import datetime
from news_recommender import NewsRecommender
import os

# Initialize recommender
@st.cache_resource
def get_recommender():
    processed_dir = "data/processed"  # directory with pre-computed vectors
    return NewsRecommender(processed_dir)

recommender = get_recommender()

st.title("News Recommendation System")

# Create tabs for different search modes
tab1, tab2 = st.tabs(["Search by Query", "Similar Articles"])

# Tab 1: Search by Query
with tab1:
    st.header("Search Articles")
    query = st.text_input("Enter your search query")
    top_k = st.slider("Number of recommendations", 1, 10, 5, key="query_slider")

    if query:
        recommendations = recommender.get_recommendations(query, top_k)
        
        if recommendations:
            for i, article in enumerate(recommendations, 1):
                with st.expander(f"{i}. {article['title']}"):
                    st.write(f"Date: {article['published_date']}")
                    st.write(f"Topic: {article['topic']}")
                    st.write(f"Domain: {article['domain']}")
                    st.write(f"Language: {article['lang']}")
                    st.write(f"Link: {article['link']}")
        else:
            st.info("No articles found.")

# Tab 2: Similar Articles
with tab2:
    st.header("Find Similar Articles")
    
    # Create a selectbox with all article titles
    titles = [article['title'] for article in recommender.articles]
    selected_title = st.selectbox("Select an article", titles)
    top_k_similar = st.slider("Number of similar articles", 1, 10, 5, key="similar_slider")

    if selected_title:
        article_idx = recommender.get_article_index(selected_title)
        similar_articles = recommender.get_similar_articles(article_idx, top_k_similar)
        
        st.subheader("Selected Article")
        with st.expander(selected_title):
            article = recommender.articles[article_idx]
            st.write(f"Date: {article['published_date']}")
            st.write(f"Topic: {article['topic']}")
            st.write(f"Domain: {article['domain']}")
            st.write(f"Language: {article['lang']}")
            st.write(f"Link: {article['link']}")
        
        st.subheader("Similar Articles")
        for i, item in enumerate(similar_articles, 1):
            article = item['article']
            similarity = item['similarity']
            with st.expander(f"{i}. {article['title']} (Similarity: {similarity:.2f})"):
                st.write(f"Date: {article['published_date']}")
                st.write(f"Topic: {article['topic']}")
                st.write(f"Domain: {article['domain']}")
                st.write(f"Language: {article['lang']}")
                st.write(f"Link: {article['link']}")

# Display dataset statistics in sidebar
with st.sidebar:
    st.header("Dataset Statistics")
    st.write(f"Total articles: {len(recommender.articles)}")
    
    # Count articles by topic
    topics = {}
    for article in recommender.articles:
        topics[article['topic']] = topics.get(article['topic'], 0) + 1
    
    st.subheader("Articles by Topic")
    for topic, count in sorted(topics.items()):
        st.write(f"{topic}: {count}") 