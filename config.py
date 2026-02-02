from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

load_dotenv()

# OpenAI API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
llm_openai = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.3
)

# Embeddings
hf_embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
