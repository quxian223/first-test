# paper_agent.py
# pip install requests feedparser schedule python-dotenv

import os
import re
import time
import json
import sqlite3
import smtplib
import schedule
import requests
import feedparser
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ======================
# 1. 配置区
# ======================

KEYWORDS = [
    "model predictive control",
    "reinforcement learning control",
    "robot control",
    "large language model control",
    "multi agent planning",
    "autonomous systems"
]

NEGATIVE_KEYWORDS = [
    "biology",
    "medical",
    "finance"
]

MAX_PAPERS_PER_SOURCE = 10
DB_PATH = "papers.db"
REPORT_DIR = "reports"

# OpenAI-compatible API
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
MAIL_TO = os.getenv("MAIL_TO", "")


# ======================
# 2. 数据库
# ======================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS papers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        abstract TEXT,
        authors TEXT,
        source TEXT,
        url TEXT UNIQUE,
        published TEXT,
        relevance_score REAL,
        summary TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()


def paper_exists(url):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM papers WHERE url = ?", (url,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def save_paper(paper):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    INSERT OR IGNORE INTO papers
    (title, abstract, authors, source, url, published, relevance_score, summary, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        paper.get("title", ""),
        paper.get("abstract", ""),
        paper.get("authors", ""),
        paper.get("source", ""),
        paper.get("url", ""),
        paper.get("published", ""),
        paper.get("relevance_score", 0),
        paper.get("summary", ""),
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


# ======================
# 3. 论文检索模块
# ======================

def search_arxiv(keyword):
    query = keyword.replace(" ", "+")
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=all:{query}"
        f"&start=0&max_results={MAX_PAPERS_PER_SOURCE}"
        "&sortBy=submittedDate&sortOrder=descending"
    )

    feed = feedparser.parse(url)
    papers = []

    for entry in feed.entries:
        papers.append({
            "title": clean_text(entry.title),
            "abstract": clean_text(entry.summary),
            "authors": ", ".join([a.name for a in entry.authors]) if hasattr(entry, "authors") else "",
            "source": "arXiv",
            "url": entry.link,
            "published": entry.published
        })

    return papers


def search_semantic_scholar(keyword):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": keyword,
        "limit": MAX_PAPERS_PER_SOURCE,
        "fields": "title,abstract,authors,year,url,publicationDate"
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
    except Exception:
        return []

    papers = []
    for item in data.get("data", []):
        authors = ", ".join([a.get("name", "") for a in item.get("authors", [])])
        papers.append({
            "title": clean_text(item.get("title", "")),
            "abstract": clean_text(item.get("abstract", "") or ""),
            "authors": authors,
            "source": "Semantic Scholar",
            "url": item.get("url", ""),
            "published": item.get("publicationDate") or str(item.get("year", ""))
        })

    return papers


def search_crossref(keyword):
    url = "https://api.crossref.org/works"
    params = {
        "query": keyword,
        "rows": MAX_PAPERS_PER_SOURCE,
        "sort": "published",
        "order": "desc"
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
    except Exception:
        return []

    papers = []
    for item in data.get("message", {}).get("items", []):
        title = item.get("title", [""])[0]
        abstract = item.get("abstract", "")
        doi = item.get("DOI", "")
        url_link = item.get("URL", "") or f"https://doi.org/{doi}"

        authors = []
        for a in item.get("author", []):
            name = f"{a.get('given', '')} {a.get('family', '')}".strip()
            if name:
                authors.append(name)

        published = ""
        date_parts = item.get("published-print") or item.get("published-online") or {}
        if "date-parts" in date_parts:
            published = "-".join(map(str, date_parts["date-parts"][0]))

        papers.append({
            "title": clean_text(title),
            "abstract": clean_text(strip_html(abstract)),
            "authors": ", ".join(authors),
            "source": "Crossref",
            "url": url_link,
            "published": published
        })

    return papers


# ======================
# 4. 筛选与评分
# ======================

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def strip_html(text):
    return re.sub(r"<.*?>", "", text or "")


def relevance_score(paper):
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

    score = 0

    for kw in KEYWORDS:
        words = kw.lower().split()
        matched = sum(1 for w in words if w in text)
        score += matched / max(len(words), 1)

    for nkw in NEGATIVE_KEYWORDS:
        if nkw.lower() in text:
            score -= 2

    if "control" in text:
        score += 1
    if "learning" in text:
        score += 0.5
    if "optimization" in text:
        score += 0.5
    if "multi-agent" in text or "multi agent" in text:
        score += 0.5

    return round(score, 2)


def is_relevant(paper, threshold=1.5):
    score = relevance_score(paper)
    paper["relevance_score"] = score
    return score >= threshold


# ======================
# 5. LLM 总结模块
# ======================

def summarize_paper(paper):
    if not LLM_API_KEY:
        return simple_summary(paper)

    prompt = f"""
你是一名量子计算方向的研究生科研助手。
请阅读下面论文信息，并用中文生成结构化总结。

论文标题：{paper.get("title")}
作者：{paper.get("authors")}
来源：{paper.get("source")}
发表时间：{paper.get("published")}
摘要：{paper.get("abstract")}

请严格按以下格式输出：

1. 研究问题：
2. 核心方法：
3. 与已有工作的区别：
4. 实验或验证方式：
5. 对量子计算方向的参考价值：
6. 阅读优先级：高 / 中 / 低
7. 一句话总结：
"""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "你擅长科研论文阅读、技术总结和研究价值判断。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        resp = requests.post(
            LLM_API_URL,
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json"
            },
            data=json.dumps(payload),
            timeout=60
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM总结失败，使用简要总结。\n错误：{e}\n\n{simple_summary(paper)}"


def simple_summary(paper):
    return f"""
1. 研究问题：{paper.get("title")}
2. 核心方法：需进一步阅读全文确认。
3. 与已有工作的区别：需进一步阅读全文确认。
4. 实验或验证方式：摘要中未完全明确。
5. 对方向的参考价值：该论文与关键词相关，建议进一步阅读。
6. 阅读优先级：中
7. 一句话总结：{paper.get("abstract", "")[:200]}...
"""


# ======================
# 6. 日报生成与提醒
# ======================

def generate_report(papers):
    os.makedirs(REPORT_DIR, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(REPORT_DIR, f"paper_report_{today}.md")

    lines = []
    lines.append(f"# 论文检索日报 - {today}\n")
    lines.append(f"共筛选出 {len(papers)} 篇相关论文。\n")

    for idx, p in enumerate(papers, 1):
        lines.append(f"## {idx}. {p['title']}\n")
        lines.append(f"- 来源：{p.get('source')}")
        lines.append(f"- 时间：{p.get('published')}")
        lines.append(f"- 作者：{p.get('authors')}")
        lines.append(f"- 链接：{p.get('url')}")
        lines.append(f"- 相关性评分：{p.get('relevance_score')}\n")
        lines.append(p.get("summary", ""))
        lines.append("\n---\n")

    content = "\n".join(lines)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return path, content


def send_email(subject, content):
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, MAIL_TO]):
        return False

    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = MAIL_TO

    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, [MAIL_TO], msg.as_string())
        return True
    except Exception as e:
        print("邮件发送失败：", e)
        return False


# ======================
# 7. 核心流程
# ======================

def run_pipeline():
    print("开始自动论文检索：", datetime.now())

    init_db()
    all_papers = []

    for kw in KEYWORDS:
        print(f"检索关键词：{kw}")

        all_papers.extend(search_arxiv(kw))
        time.sleep(1)

        all_papers.extend(search_semantic_scholar(kw))
        time.sleep(1)

        all_papers.extend(search_crossref(kw))
        time.sleep(1)

    # URL 去重
    unique = {}
    for p in all_papers:
        if p.get("url"):
            unique[p["url"]] = p

    candidate_papers = list(unique.values())

    selected = []
    for paper in candidate_papers:
        if paper_exists(paper["url"]):
            continue

        if is_relevant(paper):
            print("发现相关论文：", paper["title"])
            paper["summary"] = summarize_paper(paper)
            save_paper(paper)
            selected.append(paper)

    report_path, report_content = generate_report(selected)

    print(f"日报已生成：{report_path}")

    if selected:
        send_email(
            subject=f"论文检索日报：{len(selected)} 篇新论文",
            content=report_content
        )

    print("本轮任务完成。\n")


# ======================
# 8. 定时任务
# ======================

def start_scheduler():
    # 每天早上 8:30 执行
    schedule.every().day.at("08:30").do(run_pipeline)

    print("论文自动检索系统已启动，每天 08:30 自动运行。")
    run_pipeline()

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    start_scheduler()