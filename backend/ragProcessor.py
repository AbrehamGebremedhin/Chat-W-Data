import bs4
import os
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
        self.database = self.client.get_database(
            os.getenv("ASTRA_DB_API_ENDPOINT"))

        self.file_path = file_path

        self.user_query = user_query

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001")

        self.llm = GoogleGenerativeAI(
            model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

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

        embeddings = self.embeddings.embed_documents(str(chunks))

        return embeddings


rag = ragProcessor(file_path="OpenCog_Hyperon.pdf")
print(rag.load_vectorize_data())
