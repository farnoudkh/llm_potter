"""
Script one-shot : reconstruit l'index FAISS avec des embeddings HuggingFace locaux
(remplace les embeddings OpenAI payants).

Usage :
    python rebuild_index.py
"""
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

SOURCE_INDEX = "harry_potter_faiss_index"
TARGET_INDEX = "harry_potter_hf_index"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # supports French + English

print(f"Chargement des documents depuis {SOURCE_INDEX}/index.pkl ...")
with open(f"{SOURCE_INDEX}/index.pkl", "rb") as f:
    docstore, index_to_docstore_id = pickle.load(f)

docs = []
for doc_id in index_to_docstore_id.values():
    doc = docstore.search(doc_id)
    if doc is not None:
        docs.append(doc)

print(f"{len(docs)} documents extraits.")

print(f"Génération des embeddings avec {EMBEDDING_MODEL} (première fois : téléchargement du modèle) ...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print("Construction du nouvel index FAISS ...")
new_db = FAISS.from_documents(docs, embeddings)
new_db.save_local(TARGET_INDEX)

print(f"Index sauvegardé dans {TARGET_INDEX}/")
print("Terminé. Tu peux maintenant lancer : streamlit run script.py")
