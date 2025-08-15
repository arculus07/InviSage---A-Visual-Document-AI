import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


class EmbeddingManager:
    def __init__(self):
        """Initializes the manager and the OpenAI embeddings model."""
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    def build_vectorstore(self, docs, db_path):
        """
        Builds a new vector store in the specified directory path.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=db_path
        )

        return vectorstore