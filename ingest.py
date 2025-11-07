# we are following steps from this doc for understanding RAG  https://developers.llamaindex.ai/python/framework/understanding/rag/
# we will use llamaindex which help us in all steps from loading to querying and we use pinecone to store our docs as embeddings 
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY )
datastore_dir = os.getenv('DATASTORE_DIR', "datastore")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
) # this will be our model used for create embedding when we call VectorStoreIndex.from_document, make sure the index has same dimenion as the model 

pinecone_index = pc.Index("nano-notebook-llm-mistrall")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index) # creating a vector store within index

storage_context = StorageContext.from_defaults(vector_store=vector_store ) # get the storage context to store vectors

documents = SimpleDirectoryReader(datastore_dir).load_data() # Step 1:Loading our docs


vector_store_index = VectorStoreIndex.from_documents(
  documents,
  storage_context=storage_context
) # Step 2: creating index
