# NeuroGenius App 🧠

An academic assistant with PDF upload, chat, summarization, translation, voice input, and web search.

## Features
- Upload PDFs and query them
- Summarize documents
- Translate text
- Voice input to query
- Web search using DuckDuckGo

## Run locally

```bash
docker build -t neurogenius .
docker run -p 8501:8501 -p 8000:8000 neurogenius
