import bs4
import os
import time
import random
from langchain_community.document_loaders import WebBaseLoader, AzureAIDocumentIntelligenceLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from astrapy import DataAPIClient
from astrapy.constants import VectorMetric
from astrapy.ids import UUID
from astrapy.exceptions import InsertManyException
from dotenv import load_dotenv


class ragProcessor():
    def __init__(self, file_path, user_query=""):

        load_dotenv("../config.env")

        self.client = DataAPIClient(os.getenv(
            "ASTRA_DB_APPLICATION_TOKEN"))
        self.database = self.client.get_database_by_api_endpoint(
            os.getenv("ASTRA_DB_API_ENDPOINT"))

        self.file_path = file_path

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001")

        self.user_query = self.embeddings.embed_query(user_query)

        self.llm = GoogleGenerativeAI(
            model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

        self.collection = self.database.create_collection(
            "chat_w_document",
            dimension=1408,
            metric=VectorMetric.COSINE,
            check_exists=False,
        )

    def generate_uuid7(self):
        timestamp = int(time.time() * 1000)

        timestamp_high = (timestamp >> 28) & 0xFFFFFFFF
        timestamp_low = timestamp & 0xFFFF

        random_bits = random.getrandbits(128)

        uuid7_parts = [
            f'{timestamp_high:08x}',
            f'{timestamp_low:04x}',
            '7913',
            '89f8',
            f'{random_bits & 0xFFFFFFFFFFFF:012x}'
        ]

        uuid7_str = '-'.join(uuid7_parts)
        return uuid7_str

    def load_vectorize_data(self):
        # Load data from the file
        loader = AzureAIDocumentIntelligenceLoader(
            api_endpoint=os.getenv("ENDPOINT"), api_key=os.getenv("KEY"), file_path=self.file_path, api_model="prebuilt-layout"
        )

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=20,
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
