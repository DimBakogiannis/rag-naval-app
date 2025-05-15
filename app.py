import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the LLM and embeddings
model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.7
)

# Use a simple `StrOutputParser` to extract the answer as a string
parser = StrOutputParser()


# Create the prompt template
template = """
Answer the question based on the context below. Start with a brief introductory sentence, 
then list at least 4 main points as a simple numbered list. Each point should start with the number 
followed by the main topic. Make sure to provide at least 4 distinct points.

If you can't answer the question, reply "I don't know".

Context: {context}

Question: {question}

Format your answer like this:
[Brief introductory sentence]

1. [Point 1]: [Detailed explanation]

2. [Point 2]: [Detailed explanation]

3. [Point 3]: [Detailed explanation]

4. [Point 4]: [Detailed explanation]
"""

prompt = ChatPromptTemplate.from_template(template)

# Document Loader
# Load the transcription
loader = TextLoader("transcription.txt")
text_documents = loader.load()


# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(text_documents)

# Embedding model
embeddings = OpenAIEmbeddings()

# Vector store

index_name = YOUR_INDEX_NAME

# Debug Pinecone connection
print(f"Initializing Pinecone with index: {index_name}")
print(f"Number of documents to index: {len(documents)}")

# Initialize Pinecone with proper configuration
pinecone = PineconeVectorStore.from_documents(
    documents, 
    embeddings, 
    index_name=index_name
)

# Verify Pinecone connection
print("Pinecone initialized. Verifying connection...")
try:
    # Try to get some documents to verify connection
    test_docs = pinecone.similarity_search("test", k=1)
    print(f"Successfully connected to Pinecone. Retrieved {len(test_docs)} test documents.")
except Exception as e:
    print(f"Error connecting to Pinecone: {str(e)}")

#### Retriever
retriever = pinecone.as_retriever(
    search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
)

# Create the chain
chain = (
    RunnableParallel({
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "question": lambda x: x["question"]
    })
    | prompt
    | model
    | parser
)


#### QA Chain - Get response
def get_response(question):
    """Get response from the RAG system"""
    try:
        # Print the question being processed
        print(f"\nProcessing question: {question}")
        
        # Check what documents are being retrieved
        retrieved_docs = retriever.get_relevant_documents(question)
        print(f"\nRetrieved {len(retrieved_docs)} documents:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
        
        # Get response using the chain
        response = chain.invoke({"question": question})
        
        # Format the response
        if response and not response.startswith("I don't know"):
            # Split into paragraphs
            paragraphs = response.split('\n')
            formatted_paragraphs = []
            
            # Keep the first paragraph (intro) as is
            if paragraphs:
                formatted_paragraphs.append(paragraphs[0].strip())
            
            # Format the numbered points
            point_count = 0
            for para in paragraphs[1:]:
                if para.strip():
                    # Remove any existing formatting
                    para = para.strip()
                    # Remove any existing numbering
                    if para[0].isdigit():
                        para = para[para.find('.')+1:].strip()
                    # Remove any existing bold formatting
                    para = para.replace('**', '')
                    # Add proper formatting
                    point_count += 1
                    formatted_paragraphs.append(f"{point_count}. {para}")
            
            # Ensure we have at least 4 points
            while point_count < 4:
                point_count += 1
                formatted_paragraphs.append(f"{point_count}. [Additional point]")
            
            response = '\n\n'.join(formatted_paragraphs)
        
        print(f"\nFinal response: {response}")
        return response
    except Exception as e:
        print(f"Error details: {str(e)}")
        return f"Error: {str(e)}"

# Create the Gradio interface
demo = gr.Interface(
    fn=get_response,
    allow_flagging="never",
    inputs=gr.Textbox(
        lines=2, 
        placeholder="Ask a question about the video content...",
        label="Input Query"
    ),
    outputs=gr.Textbox(
        lines=5,
        label="Answer"
    ),
    title="YouTube Video Q&A",
    description="Ask questions about the video content and get AI-powered answers based on the video transcription.",
    theme="soft"
)

# Launch the app
demo.launch(server_name="0.0.0.0", server_port= 7860)