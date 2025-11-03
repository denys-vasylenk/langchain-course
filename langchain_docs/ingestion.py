import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap


from langchain_docs.logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False,
    chunk_size=50, retry_min_seconds=10
)

vectorestore = PineconeVectorStore(index_name="langchain-doc-assistance", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()



async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECOR STORAGE PHASE")
    log_info(f"VectorStore Indexing: preparing to add {len(documents)} documents to vector store", Colors.DARKCYAN)

    # Create batches
    batches = [documents[i: i+batch_size] for i in range(0, len(documents), batch_size)]
    log_info(f"VecoreStore Indexing: Split into {len(batches)} batches of {batch_size} documents each")

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorestore.aadd_documents(batch)
            log_success(f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)")
        except Exception as e:
            log_error(f"VectoreStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True
    
    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count succesful batches
    successful = sum(1 for result in results if result is True)
    if successful == len(batches):
        log_success(f"VectoreStore Indexing: All batches processed successfully! ({successful}/{len(batches)})")
    else:
        log_warning(f"Vecorestore Indexing: Processed {successful}/{len(batches)} batches successfully")


async def main ():

    #Access the raw Pinecone index
    index = vectorestore._index  # ⚠️ This is a private attr, but works fine

    # Delete everything in the index
    index.delete(delete_all=True)

    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATUION INGESTION STARTED")

    log_info("TavilyCrawl: Starting to crawl the documentation", 
             Colors.PURPLE)

    res = tavily_crawl.invoke({
        "url":"https://python.langchain.com/",
        "max_depth":5,
        "extract_depth":"advanced",
        "instructions":"content on ai agent"
    })
    all_docs = [Document(page_content=result['raw_content'], metadata = {"source":result['url']}) for result in res['results']]
    log_success(
        f"TavilyCrawl: Crawled {len(all_docs)} URLs from documentation site"
        )

    # spliting documents 
    log_header("Document Chunking Phase")
    log_info(f"Text Splitter: Splitting {len(all_docs)}", Colors.YELLOW)

    text_splitter= RecursiveCharacterTextSplitter(chunk_size = 4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents")

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("Pipeline Complete")
    log_success("Documentation ingestion pipeline completed successfully!")
    log_info("Summary:", Colors.BOLD)
    #log_info(f"URLs mapped: {len(site_map['results'])}")
    log_info(f"Documents extracted: {len(all_docs)}")
    log_info(f"Document chunks created: {len(splitted_docs)}")



if __name__ == "__main__":
    asyncio.run(main())