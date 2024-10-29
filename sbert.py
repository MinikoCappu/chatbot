import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Используем устройство: {device}')

with open('Конституция_без_ссылок.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

article_names = list(data.keys())
article_texts = list(data.values())

def load_models():
    model_names = ['multi-qa-mpnet-base-cos-v1', 'all-mpnet-base-v2', 'bert-base-nli-mean-tokens']
    return [SentenceTransformer(name).to(device) for name in model_names]

models = load_models()

def encode_texts(texts, models):
    embeddings = []
    for model in models:
        embedding = model.encode(texts, convert_to_tensor=True).to(device)
        embeddings.append(torch.nn.functional.normalize(embedding, p=2, dim=1))
    return torch.stack(embeddings).mean(dim=0)

article_embeddings = encode_texts(article_texts, models)

def find_similar_articles(query, article_embeddings, article_names, article_texts, top_n=3):
    if not query.strip():
        return []
    
    query_embeddings = []
    for model in models:
        query_embedding = model.encode(query, convert_to_tensor=True).to(device)
        query_embeddings.append(torch.nn.functional.normalize(query_embedding, p=2, dim=0))
    
    query_embedding = torch.stack(query_embeddings).mean(dim=0)
    
    cos_scores = util.cos_sim(query_embedding, article_embeddings)[0]
    top_indices = torch.topk(cos_scores, top_n * 2).indices
    
    similar_articles = [(article_names[i], article_texts[i], cos_scores[i].item()) for i in top_indices]
    return sorted(similar_articles, key=lambda x: x[2], reverse=True)[:top_n]

while True:
    query = input("Введите текст для поиска (или 'выход' для завершения): ").strip()
    if query.lower() in ["выход", "exit", "quit"]:
        print("Завершение программы.")
        break
    
    similar_articles = find_similar_articles(query, article_embeddings, article_names, article_texts)
    
    if similar_articles:
        for i, (article_name, article_text, score) in enumerate(similar_articles):
            print(f"{i+1}. {article_name} (схожесть: {score:.4f}):\n{article_text}\n")
    else:
        print("Запрос не дал результатов. Попробуйте другой текст.")
