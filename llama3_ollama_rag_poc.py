import os
import requests
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import docx
import numpy as np

# ==== НАСТРОЙКИ ====
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
OLLAMA_MODEL = "llama3"

# ==== ФУНКЦИИ ====
def load_documents_from_folder(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            doc = docx.Document(file_path)
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            docs.append((filename, full_text))
    return docs

def split_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def build_faiss_index(docs, model):
    embeddings = []
    metadata = []
    for filename, text in docs:
        chunks = split_text(text)
        for idx, chunk in enumerate(chunks):
            emb = model.encode(chunk)
            embeddings.append(emb)
            metadata.append((filename, idx, chunk))

    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, metadata

def search_index(query, index, model, metadata, top_k=5):
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        filename, chunk_id, chunk_text = metadata[idx]
        results.append((filename, chunk_id, chunk_text, dist))
    return results

def ask_ollama(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "Ты помощник, отвечающий кратко и по делу на русском языке."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    try:
        response = requests.post(OLLAMA_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Ошибка при вызове Ollama: {e}"

# ==== ОСНОВНОЙ КОД ====
if __name__ == "__main__":
    print("Загружаем модель эмбеддингов... (может занять немного времени)")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Загружаем документы
    folder_path = "./data"
    documents = load_documents_from_folder(folder_path)

    print("Создаём FAISS индекс...")
    index, metadata = build_faiss_index(documents, model)
    print(f"Индексация завершена. Всего фрагментов: {len(metadata)}")

    while True:
        query = input("Вопрос: ")
        if query.lower() in ["exit", "выход"]:
            break

        results = search_index(query, index, model, metadata, top_k=5)
        print("Найдено фрагментов:")
        for filename, chunk_id, _, dist in results:
            print(f" - {filename} chunk {chunk_id} (dist={dist:.4f})")

        context = "\n".join([f"[{f}#{cid}] {txt}" for f, cid, txt, _ in results])
        prompt = f"Используя контекст ниже, ответь на вопрос.\nКонтекст:\n{context}\nВопрос: {query}"

        answer = ask_ollama(prompt)
        print("\n--- Ответ: ---\n")
        print(answer)
        print("\n----------------\n")
