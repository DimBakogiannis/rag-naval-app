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
- `OpenAI Whisper` (with audio downloaded via `pytube`)
- `OpenAI GPT-4o-mini`
- `Pinecone` for vector storage
- `Gradio` for the chat interface

---

## 🚀 How to Run
1. Clone the repository, create a virtual environment, and install the required packages:

```bash
git clone https://github.com/yourusername/rag-naval-app.git
$ python3 -m venv .venv
$ source .venv/bin/activate
cd rag-naval-app
pip install -r requirements.txt
```
2. Set up environment variables. Create a `.env` file with the following variables:

```bash
OPENAI_API_KEY = [ENTER YOUR OPENAI API KEY HERE]
PINECONE_API_KEY = [ENTER YOUR PINECONE API KEY HERE]
PINECONE_API_ENV = [ENTER YOUR PINECONE API ENVIRONMENT HERE]
```
3. Run the application:
```bash
python app.py
```
---
## 🖼️ What the App Looks Like in Action
![Image](https://github.com/user-attachments/assets/86c99519-12f4-4b3d-9603-a97285e62697)
