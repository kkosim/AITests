import os, json, time, requests
from datetime import datetime

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
INPUT_FILE  = "/data/outputs/articles.jsonl"
OUTPUT_FILE = "/data/outputs/qa_dataset.jsonl"

def call_ollama(prompt: str) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def gen_questions(text: str, kind: str, n: int = 5):
    role = "пункта" if kind == "point" else "статьи"
    prompt = f"""Ты помощник по созданию датасета для обучения юридической модели.
Дан текст {role} закона:
\"\"\"{text}\"\"\"
Сгенерируй {n} разных естественных формулировок вопросов, которые человек мог бы задать, чтобы получить этот ответ.
Требования:
- без нумерации и кавычек
- на русском
- по одному вопросу в строке
- не упоминай слово "текст"

Только список вопросов:"""
    resp = call_ollama(prompt)
    lines = [l.strip().lstrip("-•").strip() for l in resp.splitlines() if l.strip()]
    # простой фильтр
    lines = [q for q in lines if len(q) > 5]
    return lines[:n]

def main():
    total = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for line in f_in:
            art = json.loads(line)
            # по статье
            qs = gen_questions(art["title"], "article", n=5)
            for q in qs:
                rec = {
                    "question": q,
                    "answer": art["title"].strip(),
                    "law_id": art["law_id"],
                    "type": "article",
                    "generated_at": datetime.now().isoformat()
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
                time.sleep(0.2)

            # по пунктам
            for p in art.get("points", []):
                qs = gen_questions(p["content"], "point", n=5)
                for q in qs:
                    rec = {
                        "question": q,
                        "answer": p["content"].strip(),
                        "law_id": p["law_id"],
                        "type": "point",
                        "generated_at": datetime.now().isoformat()
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1
                    time.sleep(0.2)
    print(f"✅ Сгенерировано Q/A: {total}. Файл: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
