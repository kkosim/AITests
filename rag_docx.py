import os
import faiss
import requests
import pickle
import hashlib
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np

# === Настройки ===
DOCS_DIR = "docs"  # Папка с .docx файлами
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"
HASH_FILE = "doc_hashes.pkl"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"

# === Хэш содержимого файла (для отслеживания изменений) ===
def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

# === Чтение .docx ===
def read_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# === Разделение текста на чанки ===
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# === Класс для поиска ===
class DocIndex:
    def __init__(self):
        print("Загружаем модель эмбеддингов...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []
        self.doc_hashes = {}

    def add_docs_from_folder(self, folder):
        new_chunks = []
        updated = False

        # Загружаем прошлые хэши
        if os.path.exists(HASH_FILE):
            with open(HASH_FILE, "rb") as f:
                self.doc_hashes = pickle.load(f)

        for file in os.listdir(folder):
            if file.lower().endswith(".docx"):
                path = os.path.join(folder, file)
                h = file_hash(path)

                # Если файла нет в хэшах или он изменился
                if file not in self.doc_hashes or self.doc_hashes[file] != h:
                    print(f"📄 Обновляем: {file}")
                    text = read_docx(path)
                    doc_chunks = split_text(text)
                    new_chunks.extend(doc_chunks)
                    self.doc_hashes[file] = h
                    updated = True

        # Если индекс уже существует — загружаем его
        if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE) and not updated:
            print("✅ Загружаем кешированный индекс...")
            with open(CHUNKS_FILE, "rb") as f:
                self.chunks = pickle.load(f)
            self.index = faiss.read_index(INDEX_FILE)
            return

        # Если были новые/изменённые файлы
        if updated or not os.path.exists(INDEX_FILE):
            # Добавляем старые чанки (если есть)
            if os.path.exists(CHUNKS_FILE):
                with open(CHUNKS_FILE, "rb") as f:
                    old_chunks = pickle.load(f)
                self.chunks = old_chunks + new_chunks
            else:
                self.chunks = new_chunks

            print("📊 Строим индекс...")
            vectors = self.embed_model.encode(self.chunks, show_progress_bar=True)
            vectors = np.array(vectors, dtype="float32")
            self.index = faiss.IndexFlatL2(vectors.shape[1])
            self.index.add(vectors)

            # Сохраняем всё
            faiss.write_index(self.index, INDEX_FILE)
            with open(CHUNKS_FILE, "wb") as f:
                pickle.dump(self.chunks, f)
            with open(HASH_FILE, "wb") as f:
                pickle.dump(self.doc_hashes, f)

    def search(self, query, top_k=3):
        q_vec = self.embed_model.encode([query])
        q_vec = np.array(q_vec, dtype="float32")
        D, I = self.index.search(q_vec, top_k)
        return [self.chunks[i] for i in I[0]]

# === Отправка в Ollama ===
def ask_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True
    }
    response = requests.post(OLLAMA_URL, json=payload, stream=True)
    answer = ""
    for line in response.iter_lines():
        if line:
            try:
                data = line.decode("utf-8")
                if '"response":"' in data:
                    part = data.split('"response":"')[1].split('"')[0]
                    answer += part
                    print(part, end="", flush=True)
            except:
                pass
    print()
    return answer

# === Основной код ===
if __name__ == "__main__":
    index = DocIndex()
    index.add_docs_from_folder(DOCS_DIR)

    while True:
        q = input("\nВопрос: ")
        if q.lower() == "exit":
            break
        relevant_chunks = index.search(q, top_k=3)
        context = "\n\n".join(relevant_chunks)
        prompt = f"Используя контекст ниже, ответь на вопрос.\n\nКонтекст:\n{context}\n\nВопрос: {q}\nОтвет:"
        ask_ollama(prompt)
