# RAG App: Exploring Ideas with Naval Ravikant using LangChain + GPT-4o-mini

This project is a Retrieval-Augmented Generation (RAG) app built with [LangChain](https://www.langchain.com/), [OpenAI GPT-4o-mini](https://openai.com/), and [Gradio](https://www.gradio.app/). Inspired by Santiago Valdarrama's YouTube RAG demo, the goal is to make long-form content, like philosophical interviews, more interactive.

🔗 [Read the full Medium article here](https://medium.com/@jimbakogiannis/building-a-retrieval-augmented-generation-rag-app-with-langchain-and-chatgpt-exploring-naval-5b1aa9eaef99)

---

## 💡 What It Does

- Transcribes a YouTube video using [OpenAI Whisper](https://openai.com/index/whisper/)
- Chunks and embeds the transcript with OpenAI Embeddings
- Stores and retrieves content using Pinecone
- Lets you ask questions to ChatGPT-4o-mini through a Gradio interface

---

## 🧪 Tech Stack

- `LangChain`
- `OpenAI Whisper` (via `youtube-transcript-api`)
- `OpenAI GPT-4o-mini`
- `Pinecone` for vector storage
- `Gradio` for the chat interface

---

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/rag-naval-app.git
cd rag-naval-app
pip install -r requirements.txt
python app/run_app.py
