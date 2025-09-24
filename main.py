import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

from dotenv import load_dotenv
load_dotenv() 

if __name__ == "__main__":
    pdf_path = "/Users/me/Desktop/vectorstor-in-memory/2210.03629v3.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embedings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorestore = FAISS.from_documents(docs, embedings)
    vectorestore.save_local("faiss_index_react")

    new_vectorestore = FAISS.load_local("faiss_index_react", embedings, allow_dangerous_deserialization=True)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")), retrieval_qa_chat_prompt
    )

    retrieval_chain = create_retrieval_chain(new_vectorestore.as_retriever(), combine_docs_chain)

    res = retrieval_chain.invoke({"input":"Give me the gist of ReAct in 3 sentences."})

    print(res["answer"])