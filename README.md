---
title: RAG KB Challenge
emoji: 🔎
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# RAG KB Challenge Space

Dense-retrieval RAG over a 50-passage technical knowledge base.

- `POST /query` with `{"question": "...", "top_k": 3}`
- Extractive answers from the best-matching sentence of the top passages
- Abstains (null answer, empty sources, confidence < 0.4) on off-topic / unanswerable queries
- Query-side prompt-injection patterns are stripped; answers are only ever quoted KB text
- CORS: `Access-Control-Allow-Origin: *`
