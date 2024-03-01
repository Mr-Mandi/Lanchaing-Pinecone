import os
import uuid
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LPinecone
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

current_directory = os.getcwd()

INDEX_NAME = ""

def connect_vector_storage() -> Pinecone:
    pinecone_api_key = ""
    pinecone_env = ""

    pinecone = Pinecone(api_key=pinecone_api_key)

    try:
        # check exists Index
        pinecone.Index(name=INDEX_NAME)
    except Exception:
        # create a Index
        pinecone.create_index(
            name="quickstart",
            dimension=8,
            metric="euclidean",
            spec=ServerlessSpec(
                cloud='',
                region=''
            )
        )

    # index_description = pinecone.describe_index(index_name)
    # print(pinecone.list_indexes())
    # print(index_description.dimension)
    # print(index_description.metric)
    # print(index_description.name)

    return pinecone


def get_information_execl() -> pd.DataFrame:
    df = pd.read_excel(f"{current_directory}/resources/Menu.xlsx")
    df.to_csv(
        f"{current_directory}/resources/Menu.csv",
        index=False
    )
    csv = pd.read_csv(f"{current_directory}/resources/Menu.csv")
    return csv


def add_info_to_vector_storage(pc: Pinecone, file_name: str):
    loader = CSVLoader(file_path=f"{current_directory}/resources/{file_name}")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    data = loader.load()
    list_values: list = []
    index = pc.Index(name=INDEX_NAME)
    for doc in data:
        list_values.append(
            {
                "id": str(uuid.uuid1()),
                "values": embeddings.embed_query(doc.page_content)
            }
        )
    index.upsert(
        vectors=list_values,
        namespace="ns1"
    )
    return data


def separate_by_chunks(file_name: str):
    loader = CSVLoader(file_path=f"{current_directory}/resources/{file_name}")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n"],
        length_function=len
    )
    chunks = text_splitter.split_documents(data)
    return chunks


def convert_document_to_embeddings(chunks: list):
    index_name = 'helpy-menu'
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    """ LPinecone.from_documents(
        chunks,
        embeddings,
        index_name=index_name
    ) """

    return embeddings


def questions_to_document(question: str, vstore):
    docs = vstore.similarity_search_with_score(question, 3)
    for doc in docs:
        print(doc[0].page_content)
        print("promedio: ", doc[1])
