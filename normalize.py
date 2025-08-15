import os
import re
import json
from datetime import datetime
from docx import Document

# === Настройки ===
INPUT_FOLDER = "docs"   # Папка с docx
OUTPUT_FILE = "articles.jsonl"  # Файл с результатом

# === Регулярки для поиска ===
ARTICLE_PATTERN = re.compile(r"(?:Статья|Article)\s+(\d+)\.?\s*(.*)", re.IGNORECASE)
POINT_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+(.*)")

def normalize_text(text):
    """Удаляет лишние пробелы и переносы"""
    return re.sub(r"\s+", " ", text).strip()

def normalize_number(num_str):
    """Удаляет ведущие нули из номера"""
    return ".".join(str(int(x)) for x in num_str.split("."))

def extract_articles_from_docx(file_path):
    """Извлекает статьи и пункты из docx"""
    doc = Document(file_path)
    articles = []
    current_article = None
    law_code = os.path.splitext(os.path.basename(file_path))[0]  # имя файла как код закона

    for para in doc.paragraphs:
        text = normalize_text(para.text)
        if not text:
            continue

        # Если нашли новую статью
        match_article = ARTICLE_PATTERN.match(text)
        if match_article:
            # Сохраняем предыдущую
            if current_article:
                articles.append(current_article)

            art_num = normalize_number(match_article.group(1))
            current_article = {
                "law_id": f"{law_code}-A{art_num}",
                "article_number": art_num,
                "title": match_article.group(2),
                "points": [],
                "source_file": os.path.basename(file_path),
                "extracted_at": datetime.now().isoformat()
            }
            continue

        # Если нашли пункт
        match_point = POINT_PATTERN.match(text)
        if match_point and current_article:
            point_num = normalize_number(match_point.group(1))
            current_article["points"].append({
                "law_id": f"{law_code}-A{current_article['article_number']}-P{point_num}",
                "point_number": point_num,
                "content": match_point.group(2)
            })
            continue

        # Если просто текст в статье
        if current_article:
            if current_article["points"]:
                # Добавляем к последнему пункту
                current_article["points"][-1]["content"] += " " + text
            else:
                # Нет пунктов — добавляем в заголовок/описание статьи
                current_article["title"] += " " + text

    # Добавляем последнюю статью
    if current_article:
        articles.append(current_article)

    return articles


def main():
    results = []

    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(".docx"):
            path = os.path.join(INPUT_FOLDER, filename)
            print(f"Обрабатываю {filename}...")
            articles = extract_articles_from_docx(path)
            results.extend(articles)

    # Сохраняем в JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Готово! Извлечено {len(results)} статей. Результат в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
