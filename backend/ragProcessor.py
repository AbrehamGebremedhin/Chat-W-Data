import bs4
import os
import time
import random
from langchain_community.document_loaders import WebBaseLoader, AzureAIDocumentIntelligenceLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from astrapy import DataAPIClient
from langchain_astradb import AstraDBVectorStore
from astrapy.constants import VectorMetric
from astrapy.ids import UUID
from astrapy.exceptions import InsertManyException
from dotenv import load_dotenv


class ragProcessor():
    def __init__(self, file_path, user_query=""):

        load_dotenv("./config.env")

        self.client = DataAPIClient(os.getenv(
            "ASTRA_DB_APPLICATION_TOKEN"))
        self.database = self.client.get_database_by_api_endpoint(
            os.getenv("ASTRA_DB_API_ENDPOINT"))

        self.file_path = file_path

        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        self.user_query = self.embeddings.embed_query(user_query)

        self.llm = Ollama(model="llama3.1")

    def load_vectorize_data(self):
        # Load data from the file
        loader = AzureAIDocumentIntelligenceLoader(
            api_endpoint=os.getenv("ENDPOINT"), api_key=os.getenv("KEY"), file_path=self.file_path, api_model="prebuilt-layout"
        )

        documents = loader.load()

        # Calculate chunk size based on document length, e.g., 10% of document length
        chunk_size = max(200, int(len(documents) * 0.1))

        # Calculate chunk overlap based on chunk size, e.g., 5% of chunk size
        chunk_overlap = int(chunk_size * 0.05)

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        chunks = text_splitter.split_documents(documents)
        print("Text splitting complete")

        vectors = []
        for chunk in chunks:
            embedding = self.embeddings.embed_documents(str(chunk))
            # Debugging line to inspect embedding
            print("Embedding: ", embedding)
            if isinstance(embedding, list) and all(isinstance(e, list) for e in embedding):
                # Assuming embed_documents returns a list of lists, take the first one
                embedding = embedding[0]
            elif not isinstance(embedding, list) or not all(isinstance(e, (int, float)) for e in embedding):
                raise ValueError("Embedding must be a list of numbers")

            vector = {
                "_id": self.generate_uuid7,
                "text": chunk,
                "$vector": embedding
            }
            vectors.append(vector)
            print("One vectorization Complete")

        print("Total Vectorization Complete")

        try:
            self.collection.insert_many(vectors)
            return True
        except InsertManyException:
            return False

    def get_answer(self):
        # Perform a similarity search
        results = self.collection.find(
            sort={"$vector": self.user_query},
            limit=10,
        )
        print("Vector search results:")
        for document in results:
            print("    ", document)


rag = ragProcessor(
    file_path="D:\Projects\Chat-W-Data\\backend\OpenCog_Hyperon.pdf", user_query="tell me about the Atomese language")
print(rag.load_vectorize_data())
print(rag.get_answer())
