from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os

openai_api_key = "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"
script_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(script_dir, 'Trichy_vec.txt')
loader = TextLoader(file_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

user_query = input("Enter your query: ")

# User response
response_template = """Your need is to come up with the response for the user shared and show me in five bullet points only.
% RESPONSE
{user_response}

YOUR RESPONSE:
"""
response_prompt_template = PromptTemplate(input_variables=["user_response"], template=response_template)
llm = OpenAI(temperature=1, openai_api_key=openai_api_key)
response_chain = LLMChain(llm=llm, prompt=response_prompt_template)

# Chain 2: Recipe
location_template = """Given the exact locations and show me near some site seeing and small stores to explore.
% LOCATION
{user_location}

YOUR RESPONSE:
"""
location_prompt_template = PromptTemplate(input_variables=["user_location"], template=location_template)
location_chain = LLMChain(llm=llm, prompt=location_prompt_template)

# Create combined chain
full_chain = SimpleSequentialChain(chains=[response_chain, location_chain], verbose=True)

# Run the chains with the retrieved documents
response_result = response_chain.run({'user_response': user_query})
response_result = location_chain.run({'user_location': response_result})

# Print the results
print(f"User Query: {user_query}")
print(f"Response Result: {response_result}")
print(f"Locations Result: {response_result}")
