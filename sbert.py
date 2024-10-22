import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Используем:{device}')

with open('Конституция.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

article_names = list(data.keys())
article_texts = list(data.values())

model = SentenceTransformer('all-mpnet-base-v2').to(device)

article_embeddings = model.encode(article_texts, convert_to_tensor=True).to(device)

def find_similar_articles(query, article_embeddings, article_names, top_n=5):
    query_embedding = model.encode(query, convert_to_tensor=True).to(device)
    cos_scores = util.cos_sim(query_embedding, article_embeddings)[0]
    top_indices = np.argpartition(-cos_scores, range(top_n))[:top_n]
    top_indices = top_indices[np.argsort(-cos_scores[top_indices])]
    return [(article_names[i], article_texts[i], cos_scores[i].item()) for i in top_indices]

while True:
    print("ВВедите текст")
    query = input()
    similar_articles = find_similar_articles(query, article_embeddings, article_names)
    for i, (article_name, article_text, score) in enumerate(similar_articles):
        print(f"{i+1}. {article_name} (схожесть: {score:.4f}):\n{article_text}\n")
