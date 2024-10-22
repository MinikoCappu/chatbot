import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('Конституция.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
article_names = list(data.keys())
article_texts = list(data.values())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(article_texts)

def find_similar_articles(query, X, article_names, article_texts, vectorizer, top_n=5):
    query_vec = vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vec, X).flatten()
    
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    return [(article_names[i], article_texts[i], similarities[i]) for i in top_indices]

while True:
    query = input()
    similar_articles = find_similar_articles(query, X, article_names, article_texts, vectorizer)

    for i, (article_name, article_text, score) in enumerate(similar_articles):
        print(f"{i+1}. {article_name} (схожесть: {score:.4f}):\n{article_text}\n")
