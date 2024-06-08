import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore

"""
{
  "clientId": "AtMJGssUCnSLJoNHQliXtiKQ",
  "secret": "apXpg5lurQ13WiudwzHccTYbDtflzNBmFKMvwakZunX2UEdzsTWF_sb8h+d,dyRAD6l6BSQIic6m1Zso9f1JF,lG3dfM,BQwECIQmW_d1vbBWj8ejH9j6tIPW.rMiKtW",
  "token": "AstraCS:AtMJGssUCnSLJoNHQliXtiKQ:5235e373839ce1e4f697db40265f2eaa5ec44e5a453e8f48faeb32ec0bb7f86f"
}
"""

os.environ["ASTRA_DB_API_ENDPOINT"] = input("ENter astra DB Api Endpoint")
os.environ["ASTRA_DB_APPLICATION_TOKEN"] = input("ENter astra DB Application Token")


# Create an embdeing instance
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Load pdf fromm data directory
loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)

documents = loader.load()

# Split documents into texts
text_spliter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
final_document = text_spliter.split_documents(documents)

print(final_document[1])

# Create Vectordb on DataStax
vstore = AstraDBVectorStore(
        collection_name = "MedBot_db",
        embedding = embeddings,
        token = os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
        api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
)

# Create embedding by inserting your document into the vectore store
inserted_ids = vstore.add_documents(final_document)
print(f"\nInserted {len(inserted_ids)} documents.")


print("\nVector DB Successfully Created!")