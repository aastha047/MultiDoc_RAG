#step 1
import os
from langchain_community.document_loaders import PyPDFLoader

def load_documents(folder_path: str):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist")

    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"📄 Loading: {filename}")
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
    return documents
#step 2
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️ Created {len(chunks)} chunks")
    return chunks
#step 3
from langchain_huggingface import HuggingFaceEmbeddings

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
#step 4
from langchain_chroma import Chroma

def create_vector_store(chunks):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory="./chroma_db",
        collection_name="rag_docs"
    )
    return vector_store

#step 5(most imp)
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def query_rag_system(query_text, vector_store):
    llm = ChatOllama(model="llama3") # Make sure you have Ollama installed and running!

    retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20}
)

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant.
        Answer ONLY using the context below.
        If the answer is not present, say "I don't know."

        Context:
        {context}

        Question:
        {question}
        """
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query_text)

#step 6
def main():
    folder_path = os.path.join(os.getcwd(), "data") # CHANGE THIS to your folder path

    if not os.path.exists("./chroma_db"):
        print("📦 No vector DB found. Creating one...")
        docs = load_documents(folder_path)
        chunks = split_text(docs)
        vector_store = create_vector_store(chunks)
        print("Vector database created")
    else:
        print("📦 Loading existing vector DB...")
        vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embedding_function,
            collection_name="rag_docs"
        )

    while True:
        query = input("\n❓ Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        print("🤔 Thinking...")
        answer = query_rag_system(query, vector_store)
        print("\n🧠 Answer:\n", answer)

if __name__ == "__main__":
    main()