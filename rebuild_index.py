import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

SOURCE_INDEX = "harry_potter_hf_index"
TARGET_INDEX = "harry_potter_mpnet_index"
EMBEDDING_MODEL = "all-mpnet-base-v2"

with open(f"{SOURCE_INDEX}/index.pkl", "rb") as f:
    docstore, index_to_docstore_id = pickle.load(f)

docs = []
for doc_id in index_to_docstore_id.values():
    doc = docstore.search(doc_id)
    if doc is not None:
        docs.append(doc)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

new_db = FAISS.from_documents(docs, embeddings)
new_db.save_local(TARGET_INDEX)
