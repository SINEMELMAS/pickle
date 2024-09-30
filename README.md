# pickle
A simple example for "pickle".
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "The sky is blue and beautiful.",
    "Love this blue and bright sky!",
    "The quick brown fox jumps over the lazy dog.",
    "A king's breakfast has sausages, ham, bacon, eggs, toast, and beans.",
    "I love green eggs, ham, sausages, and bacon!",
    "The brown fox is quick and the blue dog is lazy!",
    "The sky is very blue and the sky is very beautiful today!",
]

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

with open('tfidf_model.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)

with open('tfidf_model.pkl', 'rb') as file:
    loaded_tfidf_vectorizer = pickle.load(file)

new_text = ["The sky is blue today."]
new_text_tfidf = loaded_tfidf_vectorizer.transform(new_text)
print("New text TF-IDF values:")
print(new_text_tfidf.toarray())
feature_names = loaded_tfidf_vectorizer.get_feature_names_out()

for word, score in zip(feature_names, new_text_tfidf.toarray()[0]):
    print(f"{word}: {score}")
