import os
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
#from fastapi import FastAPI, Request, Form, Response
#from fastapi.responses import HTMLResponse
#from fastapi.templating import Jinja2Templates
#from fastapi.staticfiles import StaticFiles
#from fastapi.encoders import jsonable_encoder
#from qdrant_client import QdrantClient

local_llm = "BioMistral-7B.Q4_K_M.gguf"

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path= local_llm,
    temperature=0.3,
    max_tokens=2048,
    top_p=1,
    n_ctx= 2048
)

print("LLM Initialized....")