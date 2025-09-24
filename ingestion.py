import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
load_dotenv()



if __name__ == "__main__":
    print("Ingesting ...")
    loader = TextLoader("/Users/me/Desktop/intro-to-vector-dbs/mediumblog1.txt")
    document = loader.load()

    print("splitting ...")
    text_spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_spliter.split_documents(document)
    # for d in texts:
    #     print(len(d.page_content))
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))
    print('Done')
    