import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("GRADIO_SSR_MODE", "false")

import gradio as gr
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

KB_PATH = Path(__file__).parent / "kb.json"
KB: list[str] = json.loads(KB_PATH.read_text(encoding="utf-8"))

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

PASSAGE_EMB = model.encode(
    KB, batch_size=64, normalize_embeddings=True, convert_to_numpy=True
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
SENTENCES: list[list[str]] = [_SENT_SPLIT.split(p.strip()) for p in KB]
FLAT_SENTENCES: list[str] = [s for sents in SENTENCES for s in sents if s.strip()]
SENT_OWNER: list[int] = [i for i, sents in enumerate(SENTENCES) for s in sents if s.strip()]
SENT_EMB = model.encode(
    FLAT_SENTENCES, batch_size=128, normalize_embeddings=True, convert_to_numpy=True
)

PASSAGE_THRESHOLD = 0.34
SENTENCE_THRESHOLD = 0.30
CONF_LO, CONF_HI = 0.62, 0.95
LEXICAL_WEIGHT = 0.35

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "what", "which", "who", "whom", "whose", "when",
    "where", "why", "how", "in", "on", "of", "to", "for", "with", "by",
    "at", "from", "and", "or", "not", "it", "its", "this", "that", "these",
    "those", "many", "much", "long", "can", "must", "will", "would", "use",
}


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = set()
    for w in words:
        if w in _STOPWORDS or len(w) < 2:
            continue
        out.add(w[:4] if len(w) >= 4 else w)
    return out


def _lexical_coverage(question_terms: set[str], sentence_terms: set[str]) -> float:
    if not question_terms:
        return 0.0
    return len(question_terms & sentence_terms) / len(question_terms)


def _value_seeking(question: str) -> bool:
    return (
        question.lower().startswith(("what", "which", "how"))
        or bool(re.search(
            r"how (many|much|long|often)|what (port|threshold|size|key|rate|timeout|version)",
            question,
            re.IGNORECASE,
        ))
    )


def _asks_entity(question: str) -> bool:
    return question.lower().startswith("which")


def _fact_bonus(sentence: str, value_seeking: bool, asks_entity: bool) -> float:
    if re.search(r"\d", sentence):
        return 0.09 if value_seeking else 0.0
    if asks_entity:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*", sentence)
        if any(t[0].isupper() and len(t) >= 2 for t in tokens[1:]):
            return 0.10
    return 0.0

INJECTION_PATTERNS = re.compile(
    r"ignore (all |any |previous |prior |above )?(instructions|prompts|rules)"
    r"|disregard (all |any |previous |prior |above )?(instructions|prompts|rules)"
    r"|reveal (your )?(system )?(prompt|instructions)"
    r"|you are now|new instructions?:|system prompt:"
    r"|</?(system|instruction|assistant|user)>",
    re.IGNORECASE,
)


def sanitize(question: str) -> str:
    cleaned = INJECTION_PATTERNS.sub("", question)
    return cleaned.strip()


class QueryIn(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)


class QueryOut(BaseModel):
    answer: str | None
    sources: list[int]
    scores: list[float]
    confidence: float


app = FastAPI(title="RAG KB Challenge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AlwaysCors(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers.setdefault("Access-Control-Allow-Origin", "*")
        response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
        return response


app.add_middleware(AlwaysCors)


def clip(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


@app.get("/health")
def health():
    return {"status": "ok", "kb_size": len(KB)}


def run_query(question: str, top_k: int = 3) -> dict:
    raw_question = question.strip()
    if not raw_question or len(raw_question) > 2000:
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.1}

    question = sanitize(raw_question)
    if not question:
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.1}

    q = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
    q_terms = _terms(question)
    passage_scores = PASSAGE_EMB @ q
    order = np.argsort(-passage_scores)[: max(top_k, 3)]
    top_passages = [int(i) for i in order]
    best_passage_score = float(passage_scores[top_passages[0]])

    if best_passage_score < PASSAGE_THRESHOLD:
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.15}

    top_set = set(top_passages)
    candidate_sent_ids = [j for j, owner in enumerate(SENT_OWNER) if owner in top_set]
    sent_cosines = SENT_EMB[candidate_sent_ids] @ q
    value_seeking = _value_seeking(question)
    asks_entity = _asks_entity(question)
    combined = []
    for j, cos in zip(candidate_sent_ids, sent_cosines):
        lex = _lexical_coverage(q_terms, _terms(FLAT_SENTENCES[j]))
        combined.append(float(cos) + LEXICAL_WEIGHT * lex + _fact_bonus(FLAT_SENTENCES[j], value_seeking, asks_entity))
    best_local = int(np.argmax(combined))
    best_sent_id = candidate_sent_ids[best_local]
    best_sent_score = float(sent_cosines[best_local])

    if best_sent_score < SENTENCE_THRESHOLD:
        return {
            "answer": None,
            "sources": top_passages[:top_k],
            "scores": [clip(passage_scores[i]) for i in top_passages[:top_k]],
            "confidence": 0.35,
        }

    answer = FLAT_SENTENCES[best_sent_id].strip()
    confidence = clip(
        CONF_LO + (best_sent_score - SENTENCE_THRESHOLD)
        / max(1e-6, (0.75 - SENTENCE_THRESHOLD))
        * (CONF_HI - CONF_LO)
    )
    confidence = min(confidence, CONF_HI)

    return {
        "answer": answer,
        "sources": top_passages[:top_k],
        "scores": [clip(passage_scores[i]) for i in top_passages[:top_k]],
        "confidence": round(confidence, 4),
    }


@app.post("/query", response_model=QueryOut)
def query(payload: QueryIn) -> QueryOut:
    result = run_query(payload.question, payload.top_k)
    return QueryOut(**result)


def _ui_query(question: str, top_k: int) -> str:
    result = run_query(question, int(top_k))
    return json.dumps(result, indent=2)


def _build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG KB Challenge") as demo:
        gr.Markdown(
            "# RAG KB Challenge\n"
            "Dense-retrieval RAG over a 50-passage technical knowledge base.\n\n"
            "Programmatic API: `POST /query` with `{\"question\": str, \"top_k\": int}`."
        )
        with gr.Row():
            q_in = gr.Textbox(label="Question", placeholder="What port does the Redis server bind to by default?")
            k_in = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="top_k")
        btn = gr.Button("Ask", variant="primary")
        out = gr.Code(label="Response", language="json")
        btn.click(_ui_query, inputs=[q_in, k_in], outputs=out)
        q_in.submit(_ui_query, inputs=[q_in, k_in], outputs=out)
    return demo


app = gr.mount_gradio_app(app, _build_ui(), path="/ui", ssr_mode=False)


@app.get("/")
def index():
    return {
        "service": "RAG KB Challenge",
        "endpoints": {
            "POST /query": {"question": "string", "top_k": 3},
            "GET /health": None,
        },
        "ui": "/ui",
    }


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", "7860")))
    uvicorn.run(app, host="0.0.0.0", port=port)
