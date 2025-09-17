# Finance Buddy (RAG) — OpenStax Principles of Finance

A Retrieval‑Augmented Generation (RAG) chatbot that helps **first‑time earners** build a simple, sustainable budget. It grounds answers in the **OpenStax _Principles of Finance_** textbook (free, openly‑licensed) and an optional mini‑FAQ.

> **Not financial advice.** Educational use only.

---

## 🧠 Problem Statement
**Finance Buddy** is a lightweight chatbot for **new earners** who want clear, actionable guidance on **budgeting, saving, and paying down debt**. It combines a trusted open textbook (OpenStax _Principles of Finance_) with a simple RAG pipeline to deliver concise answers and starter plans (e.g., 50/30/20) with transparent citations.

- **Target users:** First‑time earners and students starting to manage paychecks
- **Core functionality:** Natural‑language Q&A with source‑backed explanations, quick budget examples, and practical steps (automate saving, emergency fund, debt prioritization).

---

## 🧰 Tools & Frameworks
- **Python 3.10+**
- **OpenAI Python SDK (v1.x)** — fast, reliable embeddings (`text-embedding-3-small`) and chat models (e.g., `gpt-4.1-mini`).
- **pypdf** — simple, dependency‑light PDF text extraction.
- **NumPy** — cosine similarity + vector math.
- **tqdm** — progress bars for ingestion.
- **rich** — friendly CLI output.

No heavyweight vector DB required; we store normalized embeddings in a local `.npz` with JSONL metadata for clarity and portability.

---

## 📁 Project Structure
```
finance-literacy-bot/
├─ chatbot.py              # CLI chatbot (retrieval + LLM)
├─ ingest.py               # build local vector index from PDF + FAQs
├─ requirements.txt
├─ LICENSE (MIT)
├─ .gitignore
├─ data/
│  ├─ README.txt           # where to place the PDF
│  └─ sample_faqs.json     # small optional FAQ set
└─ index/
   └─ (built artifacts) vectors.npz, meta.jsonl
```

---

## 📥 Get the OpenStax PDF
1) Visit the official page and click **Download a PDF**:
```
https://openstax.org/details/books/principles-finance
```
2) Save it to `data/openstax_principles_of_finance.pdf` (exact name).

OpenStax books are typically licensed under **CC BY 4.0**. Always verify the license on the book page.

---

## 🔧 Setup
```bash
# 1) (optional) create venv
python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) install deps
pip install -r requirements.txt

# 3) set your API key
export OPENAI_API_KEY=sk-...   # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
```

---

## 🏗️ Build the Index (RAG)
```bash
# with FAQs (recommended for pragmatic tips)
python ingest.py --pdf data/openstax_principles_of_finance.pdf --faqs data/sample_faqs.json

# or just the textbook
python ingest.py --pdf data/openstax_principles_of_finance.pdf
```

This creates:
```
index/vectors.npz   # normalized embeddings (float32)
index/meta.jsonl    # one JSON object per chunk with: source, page, text, etc.
```

---

## 💬 Run the Chatbot
Interactive REPL:
```bash
python chatbot.py
```
One‑shot question:
```bash
python chatbot.py --ask "How do I build a starter budget on $3,000/mo take‑home?"
```

**What you’ll get:** short, actionable guidance with citations like `(OpenStax, p. 12)` where possible.

---

## 🧪 Example Prompts
- “Show a 50/30/20 starter budget and how to automate saving.”
- “I have a $1,000 emergency. Should I use savings or a credit card?”
- “How do I prioritize debts if my rent is $1,100 and net pay is $2,800?”

---

## 🧾 GitHub: Repo Initialization
```bash
# from inside finance-literacy-bot/
git init
git add .
git commit -m "Initial commit: Finance Buddy (RAG)"
# Option A: with GitHub CLI (recommended)
gh repo create finance-literacy-bot --public --source=. --remote=origin --push
# Option B: create an empty repo on GitHub UI, then:
git remote add origin https://github.com/<your-username>/finance-literacy-bot.git
git push -u origin main
```

**Repo description (suggested):**
> A RAG chatbot for first‑time earners, grounded in OpenStax _Principles of Finance_. Answers budgeting, saving, and debt questions with citeable context.

**License:** MIT (included).

---

## 🧼 Notes
- This project intentionally avoids heavy databases to stay classroom‑friendly.
- For larger corpora or multi‑PDF ingestion, consider adding a vector DB (e.g., Chroma/PGVector) and a sitemap/URL loader.
- Educational only; not financial advice.
