import json
import requests
from datetime import datetime
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:8b"  # можно заменить на свою модель

INPUT_FILE = "articles.jsonl"
OUTPUT_FILE = "qa_dataset.jsonl"

def call_ollama(prompt):
    """Отправка запроса в Ollama для генерации текста"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"❌ Ошибка Ollama: {e}")
        return None

def generate_questions(article_text, is_point=False):
    """Генерирует список вопросов по тексту"""
    if is_point:
        prompt = f"""Ты помощник по созданию обучающего датасета.
Дан текст пункта закона:
\"\"\"{article_text}\"\"\"
Сгенерируй 5 разных формулировок вопросов, которые человек мог бы задать, чтобы получить этот ответ.
Пиши только список вопросов, без лишнего текста."""
    else:
        prompt = f"""Ты помощник по созданию обучающего датасета.
Дан текст статьи закона:
\"\"\"{article_text}\"\"\"
Сгенерируй 5 разных формулировок вопросов, которые человек мог бы задать, чтобы получить этот ответ.
Пиши только список вопросов, без лишнего текста."""

    resp = call_ollama(prompt)
    if not resp:
        return []
    
    # Разбиваем по строкам, удаляя нумерацию
    questions = []
    for line in resp.split("\n"):
        q = line.strip().lstrip("1234567890. ").strip()
        if len(q) > 3:
            questions.append(q)
    return questions

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        articles = [json.loads(line) for line in f]

    total_pairs = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for art in articles:
            # Вопросы по статье
            q_list = generate_questions(art["title"], is_point=False)
            for q in q_list:
                out_f.write(json.dumps({
                    "question": q,
                    "answer": art["title"].strip(),
                    "law_id": art["law_id"],
                    "type": "article",
                    "generated_at": datetime.now().isoformat()
                }, ensure_ascii=False) + "\n")
                total_pairs += 1
                time.sleep(0.5)  # чтобы не перегружать Ollama

            # Вопросы по пунктам
            for p in art.get("points", []):
                q_list = generate_questions(p["content"], is_point=True)
                for q in q_list:
                    out_f.write(json.dumps({
                        "question": q,
                        "answer": p["content"].strip(),
                        "law_id": p["law_id"],
                        "type": "point",
                        "generated_at": datetime.now().isoformat()
                    }, ensure_ascii=False) + "\n")
                    total_pairs += 1
                    time.sleep(0.5)

    print(f"✅ Сгенерировано {total_pairs} Q/A пар. Результат в {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
