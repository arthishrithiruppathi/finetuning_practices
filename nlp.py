import os
import warnings
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  
warnings.filterwarnings("ignore")

import numpy as np
from transformers import pipeline
from transformers.utils import logging
logging.set_verbosity_error()                          

# One review used as the input text for several tasks
review = ("I bought this laptop from Amazon in Chennai last month. The battery life "
          "is excellent and the screen is very sharp, but the keyboard feels cheap "
          "and the fan gets loud when I run heavy software. Delivery was quick.")


# 1. Feature Extraction - turns text into numbers (a vector)
print("\n1. Feature Extraction")
fe = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
vector = np.array(fe("Transformers is an awesome library!")[0]).mean(axis=0)
print("Vector size:", vector.shape)            
print("First 5 numbers:", np.round(vector[:5], 3))


# 2. Fill-Mask - guesses the hidden word
print("\n2. Fill-Mask")
fill = pipeline("fill-mask", model="distilroberta-base")
print("Sentence: Data analysts use SQL and <mask> to analyze data.")
for guess in fill("Data analysts use SQL and <mask> to analyze data.")[:3]:
    print(guess["token_str"].strip(), round(guess["score"], 2))


# 3. Question Answering - finds the answer inside a given text
print("\n3. Question Answering")
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
result = qa(question="What is wrong with the keyboard?", context=review)
print("Question: What is wrong with the keyboard?")
print("Answer:", result["answer"], round(result["score"], 2))


# 4. Summarization - shortens a text
print("\n4. Summarization")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
print(summarizer(review, max_length=40, min_length=15, do_sample=False)[0]["summary_text"].strip())


# 5. Text Classification - sentence logic (NLI) and positive/negative sentiment
print("\n5. Text Classification")
classifier = pipeline("text-classification", model="roberta-large-mnli")
text = "A soccer game with multiple males playing. Some men are playing a sport."
result = classifier(text)[0]
print("NLI:", text)
print(result["label"], round(result["score"], 2))
sentiment = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
text = "The battery life is excellent and the screen is very sharp."
result = sentiment(text)[0]
print("Sentiment:", text)
print(result["label"], round(result["score"], 2))


# 6. Text Generation - continues a sentence
print("\n6. Text Generation")
generator = pipeline("text-generation", model="gpt2")
prompt = "The best thing about learning data analysis is"
print(generator(prompt, max_new_tokens=30, pad_token_id=50256)[0]["generated_text"])


# 7. Token Classification - finds names of organizations, places, people
print("\n7. Token Classification")
ner = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")
for item in ner(review):
    print(item["word"], item["entity_group"], round(float(item["score"]), 2))

#8.Translation
print("\n8. Translation")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
print(translator("I love learning Python programming."))

# 9. Sentence Similarity - measures how close two sentences are in meaning
print("\n9. Sentence Similarity")
sim = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

def embed(text):
    return np.array(sim(text)[0]).mean(axis=0)

def cosine(x, y):
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

s1 = "The battery life is excellent."
s2 = "The laptop lasts a long time on one charge."
s3 = "The delivery was very late."
print("Similarity 1 and 2:", round(cosine(embed(s1), embed(s2)), 2))
print("Similarity 1 and 3:", round(cosine(embed(s1), embed(s3)), 2))


# 10. Table Question Answering - answers questions from a table
print("\n10. Table Question Answering")
table = pd.DataFrame({
    "Name": ["Arun", "Divya", "Karthik"],
    "Course": ["Python", "SQL", "Python"],
    "Marks": ["85", "92", "78"],
})
table = table.astype(object)
tqa = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq")
result = tqa(table=table, query="Which student has the highest marks?")
print("Question: Which student has the highest marks?")
print("Answer:", result["answer"])


# 11. Text Ranking - sorts texts by how well they match a question
print("\n11. Text Ranking")
rank_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
rank_tokenizer = AutoTokenizer.from_pretrained(rank_name)
rank_model = AutoModelForSequenceClassification.from_pretrained(rank_name)
query = "How do I analyze data with Python?"
docs = [
    "Pandas is a Python library used to clean and analyze data.",
    "The Eiffel Tower is located in Paris.",
    "SQL is used to query data stored in databases.",
]
inputs = rank_tokenizer([query] * len(docs), docs, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    scores = rank_model(**inputs).logits.squeeze(-1)
print("Query:", query)
for doc, score in sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True):
    print(round(score, 2), doc)


# 12. Zero-Shot Classification - classifies text into labels it was never trained on
print("\n12. Zero-Shot Classification")
zero = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
text = "The battery life is excellent but the keyboard feels cheap."
labels = ["product review", "sports news", "weather report"]
result = zero(text, candidate_labels=labels)
print("Text:", text)
for label, score in zip(result["labels"], result["scores"]):
    print(label, round(score, 2))