import json
import re
from striprtf.striprtf import rtf_to_text

def extract_text_from_rtf(rtf_file_path):
    with open(rtf_file_path, 'r', encoding='windows-1251') as file:
        rtf_content = file.read()

    plain_text = rtf_to_text(rtf_content)
    plain_text = plain_text.replace('\n', ' ')
    return plain_text

def parse_articles_to_json(text):
    article_pattern = re.compile(r'Статья\s\d+')
    articles = article_pattern.split(text)
    articles = [a.strip() for a in articles if a.strip()]
    articles_json = {}
    for i, article in enumerate(articles, start=1):
        articles_json[f"Статья {i}"] = article
    return articles_json
def rtf_to_json(rtf_file_path, json_file_path):
    text = extract_text_from_rtf(rtf_file_path)
    articles_json = parse_articles_to_json(text)
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(articles_json, json_file, ensure_ascii=False, indent=4)

rtf_file_path = 'consti.rtf'
json_file_path = 'Конституция.json'
rtf_to_json(rtf_file_path, json_file_path)
