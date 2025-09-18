
#!/usr/bin/env python3
"""
chatbot.py
----------
A CLI RAG assistant for first-time earners, grounded in OpenStax "Principles of Finance"
and (optionally) a small FAQ file. It retrieves the top-K chunks and asks an LLM to
answer with short, practical guidance + citations (page numbers when available).

Usage:
  # First, run ingestion to build the index:
  #   python ingest.py --pdf data/openstax_principles_of_finance.pdf --faqs data/sample_faqs.json
  #
  # Then run the chatbot:
  #   python chatbot.py                         # interactive REPL
  #   python chatbot.py --ask "How do I make a starter budget?"

Requires OPENAI_API_KEY in env.
"""
import os, argparse, json, sys, math, textwrap
from typing import List, Dict, Any, Tuple
import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# OpenAI SDK v1.x
try:
    from openai import OpenAI
except ImportError as e:
    raise SystemExit("Missing dependency 'openai'. Run: pip install -r requirements.txt") from e

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4.1-mini"  # change to gpt-4o-mini if you prefer
TOP_K = 4
MAX_CONTEXT_CHARS = 4500  # keep context concise

console = Console()

def load_index(index_dir: str = "index") -> Tuple[np.ndarray, list]:
    vec_path = os.path.join(index_dir, "vectors.npz")
    meta_path = os.path.join(index_dir, "meta.jsonl")
    if not (os.path.exists(vec_path) and os.path.exists(meta_path)):
        raise SystemExit("Missing index files. Run: python ingest.py --pdf data/openstax_principles_of_finance.pdf")

    arr = np.load(vec_path)
    vectors = arr["vectors"].astype(np.float32)
    metas = []
    with open(meta_path, "r") as f:
        for line in f:
            metas.append(json.loads(line))
    if len(metas) != len(vectors):
        raise SystemExit("Index corrupted: vectors and metadata length mismatch.")
    return vectors, metas

def cosine_top_k(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int = TOP_K) -> List[int]:
    # doc_vecs are already normalized in ingestion
    q = query_vec / max(np.linalg.norm(query_vec), 1e-8)
    sims = doc_vecs @ q
    idx = np.argpartition(-sims, min(k, len(sims)-1))[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx.tolist()

def embed(client: "OpenAI", text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array(resp.data[0].embedding, dtype=np.float32)

def build_context(passages: List[str], metadatas: List[Dict[str,Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    items = []
    total = 0
    for txt, meta in zip(passages, metadatas):
        tag = f"(source={meta.get('source')}, page={meta.get('page')})"
        chunk = f"### Passage {len(items)+1} {tag}\n{txt.strip()}\n"
        if total + len(chunk) > max_chars and items:
            break
        items.append(chunk)
        total += len(chunk)
    return "\n".join(items)

def ask_llm(client: "OpenAI", question: str, context: str) -> str:
    system = (
        "You are a helpful financial literacy assistant for first-time earners. "
        "Use ONLY the provided context (from OpenStax 'Principles of Finance' and optional FAQs). "
        "Answer succinctly with practical steps. When you make a claim, cite the relevant page(s) "
        "as (OpenStax, p.X) if available. If the answer is not in the context, say you don't know. "
        "This is educational information, not financial advice."
    )
    user = (
        f"Question: {question}\n\n"
        f"Context begins.\n{context}\nContext ends."
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":user}
            ],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # Fallback to a smaller model if configured one fails
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":system},
                    {"role":"user","content":user}
                ],
                temperature=0.2
            )
            return resp.choices[0].message.content.strip()
        except Exception as e2:
            raise SystemExit(f"LLM call failed: {e2}")

def interactive_loop(client: "OpenAI", doc_vecs: np.ndarray, metas: list):
    console.print(Panel.fit(
        "[bold]Welcome to Finance Buddy (RAG)[/bold]\n"
        "Ask about budgeting, saving, or debt. Type 'quit' to exit."
    ))
    while True:
        q = Prompt.ask("[bold green]You[/bold green]")
        if q.strip().lower() in {"quit","exit","q"}:
            console.print(":wave: Bye!")
            break
        if not q.strip():
            continue
        qvec = embed(client, q)
        top_idx = cosine_top_k(qvec, doc_vecs, TOP_K)
        passages = []
        sel_metas = []
        for i in top_idx:
            m = metas[i]
            txt = m.get("text","")
            passages.append(txt)
            sel_metas.append(m)
        context = build_context(passages, sel_metas)
        ans = ask_llm(client, q, context)
        console.print(Panel(ans, title="Answer", expand=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", default=None, help="Ask a single question and exit")
    ap.add_argument("--index", default="index", help="Index directory")
    args = ap.parse_args()

    client = OpenAI()
    doc_vecs, metas = load_index(args.index)
    # Sanity check: ensure text present
    if not metas or "text" not in metas[0]:
        raise SystemExit("Index missing text content. Re-run ingest.py from this repository version.")

    if args.ask:
        qvec = embed(client, args.ask)
        top_idx = cosine_top_k(qvec, doc_vecs, TOP_K)
        passages = [metas[i].get("text","") for i in top_idx]
        sel_metas = [metas[i] for i in top_idx]
        context = build_context(passages, sel_metas)
        ans = ask_llm(client, args.ask, context)
        print(ans)
    else:
        interactive_loop(client, doc_vecs, metas)

if __name__ == "__main__":
    main()
