from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from getpass import getpass
import os
import csv
from openai import RateLimitError
import time
# Get OpenAI API key
openai_api_key = "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"
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load documents
file_path = os.path.join(script_dir, 'Trichy_vec.txt')
loader = TextLoader(file_path)
documents = loader.load()

# Split documents into texts
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Create FAISS index
db = FAISS.from_documents(texts, embeddings)

# Create retriever
retriever = db.as_retriever()

user_query = input("Enter your query: ")
retrieved_docs = retriever.get_relevant_documents(user_query)
llm = OpenAI(temperature=1, openai_api_key=openai_api_key)

# chain 1 user_response template
response_template = """Your need is to come up with the user response and give relevant answers only.
% USER QUERY
{user_query}

YOUR RESPONSE:
% RELEVANT ANSWERS
{relevant_answers}
"""

response_prompt_template = PromptTemplate(input_variables=["user_query", "relevant_answers"], template=response_template)
response_chain = LLMChain(llm=llm, prompt=response_prompt_template)
# chain 1
response_result = response_chain.run(retrieved_docs)

# Chain 2: Location Chain
location_template = """Given the user's plan to visit colleges in Trichy, here are some relevant details:
% RESPONSE
{user_response}

YOUR RESPONSE:
% SIMILAR ANSWERS
{similar_answers}
"""
location_prompt_template = PromptTemplate(input_variables=["user_response", "similar_answers"], template=location_template)

response_chain = LLMChain(llm=llm, prompt=response_prompt_template)
location_chain = LLMChain(llm=llm, prompt=location_prompt_template)

# Run the chains with the retrieved documents
response_result = response_chain.run({'user_query': user_query, 'relevant_answers': retrieved_docs})
response_result = location_chain.run({'user_response': response_result})

# Print the results
print(f"User Query: {user_query}")
print(f"Response Result: {response_result}")
print(f"Locations Result: {response_result}")



# Chain 3: show near shops
# explore_template = """Given the nearby shop the user shared, show me if the shops are for men, women, or kids.
# % near_places
# {near_places}

# YOUR RESPONSE:
# """
# near_place_prompt_template = PromptTemplate(input_variables=["near_places"], template=explore_template)
# near_place_chain = LLMChain(llm=llm, prompt=near_place_prompt_template)

# Combined Chain
# full_chain = SimpleSequentialChain(chains=[response_chain, location_chain, near_place_chain], verbose=True)

# Get user input for the query
# user_query = input("Enter your query: ")

# # Retrieve relevant documents from the FAISS index
# retrieved_docs = retriever.get_relevant_documents(user_query)

# # Run the chains with the retrieved documents
# response_result = response_chain.run(retrieved_docs)
# location_result = location_chain.run(retrieved_docs)
# near_place_result = near_place_chain.run(retrieved_docs)

# # Print the results
# print(f"User Query: {user_query}")
# print(f"Location Result: {response_result}")
# print(f"Recipe Result: {location_result}")
# print(f"Ingredient Cost Result: {near_place_result}")
