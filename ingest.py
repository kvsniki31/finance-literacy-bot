
#!/usr/bin/env python3
"""
ingest.py
---------
Build a lightweight local vector index from the OpenStax PDF and optional FAQs.
- Extracts text per page using pypdf
- Chunks text into ~900-char passages with 120-char overlap
- Embeds with OpenAI text-embedding-3-small
- Saves to ./index/{vectors.npy, meta.jsonl}

Usage:
  python ingest.py --pdf data/openstax_principles_of_finance.pdf
Optional:
  --faqs data/sample_faqs.json
  --chunk 900 --overlap 120
Requires OPENAI_API_KEY in env.
"""
import os, json, argparse, math, time
from typing import List, Dict, Any
from pypdf import PdfReader
import numpy as np
from tqdm import tqdm

# OpenAI SDK v1.x
try:
    from openai import OpenAI
except ImportError as e:
    raise SystemExit("Missing dependency 'openai'. Run: pip install -r requirements.txt") from e

EMBED_MODEL = "text-embedding-3-small"

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    # Normalize whitespace
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks

def embed_texts(client: "OpenAI", texts: List[str], batch_size: int = 128) -> np.ndarray:
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([item.embedding for item in resp.data])
        time.sleep(0.1)  # gentle pacing
    return np.array(vecs, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to OpenStax Principles of Finance PDF")
    ap.add_argument("--faqs", default=None, help="Optional JSON file with [{'question','answer'}]")
    ap.add_argument("--chunk", type=int, default=900)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--outdir", default="index")
    args = ap.parse_args()

    client = OpenAI()

    passages: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    # PDF ingestion
    if not os.path.exists(args.pdf):
        raise SystemExit(f"PDF not found: {args.pdf}")
    reader = PdfReader(args.pdf)
    for pageno in tqdm(range(len(reader.pages)), desc="Reading PDF"):
        page = reader.pages[pageno]
        text = page.extract_text() or ""
        chunks = chunk_text(text, args.chunk, args.overlap)
        for idx, ch in enumerate(chunks):
            passages.append(ch)
            metadatas.append({
                "source": os.path.basename(args.pdf),
                "type": "pdf",
                "page": pageno + 1,
                "chunk_idx": idx,
                "text": ch
            })

    # Optional FAQs
    if args.faqs and os.path.exists(args.faqs):
        with open(args.faqs, "r") as f:
            items = json.load(f)
        for i, item in enumerate(items):
            q = item.get("question","").strip()
            a = item.get("answer","").strip()
            if q and a:
                passages.append(f"Q: {q}\nA: {a}")
                metadatas.append({
                    "source": os.path.basename(args.faqs),
                    "type": "faq",
                    "page": None,
                    "chunk_idx": i,
                    "text": f"Q: {q}\nA: {a}"
                })

    if not passages:
        raise SystemExit("No passages extracted. Check your inputs.")

    # Embeddings
    vecs = embed_texts(client, passages)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
    vecs_normed = vecs / norms

    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, "vectors.npz"),
             vectors=vecs_normed.astype(np.float32))
    with open(os.path.join(args.outdir, "meta.jsonl"), "w") as f:
        for m in metadatas:
            f.write(json.dumps(m) + "\n")

    print(f"✅ Indexed {len(passages)} chunks → {args.outdir}/vectors.npz and meta.jsonl")

if __name__ == "__main__":
    main()
