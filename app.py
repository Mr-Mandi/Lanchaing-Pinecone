import os
from langchain_pinecone import PineconeVectorStore

from utils import (
    connect_vector_storage,
    get_information_execl,
    add_info_to_vector_storage,
    separate_by_chunks,
    convert_document_to_embeddings,
    questions_to_document
)

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

if __name__ == "__main__":
    pc = connect_vector_storage()
    """documents = add_info_to_vector_storage(
        pc=pc,
        file_name="Menu.csv"
    )"""
    chunks = separate_by_chunks(file_name="Menu.csv")
    embeddings = convert_document_to_embeddings(chunks=chunks)
    index_name = ''
    index = pc.Index(name=index_name)
    text_field = "text"
    vstore = PineconeVectorStore(
                index,
                embeddings,
                text_field
    )
    text = ""
    while text != "exit":
        text = input("que deseas ?: ")
        questions_to_document(question=text, vstore=vstore)
