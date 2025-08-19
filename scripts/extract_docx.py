import os, re, json
from datetime import datetime
from docx import Document

INPUT_FOLDER = "/data/docs"
OUTPUT_FILE = "/data/outputs/articles.jsonl"

ARTICLE_PATTERN = re.compile(r"(?:Статья|Article)\s+(\d+)\.?\s*(.*)", re.IGNORECASE)
POINT_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+(.*)")

def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()

def normalize_number(num_str: str) -> str:
    return ".".join(str(int(x)) for x in num_str.split("."))

def extract_articles_from_docx(path: str):
    doc = Document(path)
    articles = []
    current = None
    law_code = os.path.splitext(os.path.basename(path))[0]

    for para in doc.paragraphs:
        text = normalize_text(para.text)
        if not text:
            continue

        m_art = ARTICLE_PATTERN.match(text)
        if m_art:
            if current:
                articles.append(current)
            art_num = normalize_number(m_art.group(1))
            current = {
                "law_id": f"{law_code}-A{art_num}",
                "article_number": art_num,
                "title": m_art.group(2),
                "points": [],
                "source_file": os.path.basename(path),
                "extracted_at": datetime.now().isoformat()
            }
            continue

        m_p = POINT_PATTERN.match(text)
        if m_p and current:
            pnum = normalize_number(m_p.group(1))
            current["points"].append({
                "law_id": f"{law_code}-A{current['article_number']}-P{pnum}",
                "point_number": pnum,
                "content": m_p.group(2)
            })
            continue

        if current:
            if current["points"]:
                current["points"][-1]["content"] += " " + text
            else:
                current["title"] += " " + text

    if current:
        articles.append(current)
    return articles

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    total = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for fname in os.listdir(INPUT_FOLDER):
            if fname.lower().endswith(".docx"):
                path = os.path.join(INPUT_FOLDER, fname)
                print("Обрабатываю:", fname)
                arts = extract_articles_from_docx(path)
                for a in arts:
                    out.write(json.dumps(a, ensure_ascii=False) + "\n")
                    total += 1
    print(f"✅ Готово: {total} статей. Файл: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
