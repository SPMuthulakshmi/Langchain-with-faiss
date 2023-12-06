from langchain.chains import SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load documents
file_path = os.path.join(script_dir, 'Trichy_vec.txt')
loader = TextLoader(file_path)
documents = loader.load()

# Extract text content from documents using a generic approach
texts = [str(doc) for doc in documents]

# Create sentence embeddings using SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # You can choose another model from sentence-transformers
embeddings = model.encode(texts, convert_to_tensor=True).numpy()

# Create FAISS index
db = faiss.IndexFlatL2(embeddings.shape[1])
db.add(embeddings)

# Get user input for the query
user_query = input("Enter your query: ")

# Embed user query
user_query_embedding = model.encode([user_query], convert_to_tensor=True).numpy()

# Retrieve relevant documents from the FAISS index
_, retrieved_docs = db.search(user_query_embedding, k=5)

# 1 response template
response_template = """Your need is to come up with the user response and give relevant answers only.
% USER QUERY
{user_query}
"""
response_prompt_template = PromptTemplate(input_variables=["user_query"], template=response_template)
response_chain = SimpleSequentialChain(chains=[(response_prompt_template, retrieved_docs)], verbose=True)


# 2 location template
location_template = """Given the user's plan to visit colleges in Trichy, here are some relevant details:
% RESPONSE
{user_response}
"""

response_prompt_template = PromptTemplate(input_variables=["user_response"], template=location_template)
location_chain = SimpleSequentialChain(chains=[(response_prompt_template, retrieved_docs)], verbose=True)

# Create response chain
full_chain = SimpleSequentialChain(chains=[response_chain, location_chain], verbose=True)
# Run the chains with the retrieved documents
response_result = response_chain.run(user_query=user_query)
response_result = location_chain.run(user_response=user_query)

# Print the results
print(f"User Query: {user_query}")
print(f"Response Result: {response_result}")
print(f"Locations Result: {response_result}")
