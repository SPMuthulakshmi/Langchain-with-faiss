#langchain and summarizer
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from getpass import getpass
import os
from transformers import pipeline

openai_api_key = "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"
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load documents
file_path = os.path.join(script_dir, 'Trichy.txt')

# Specify the encoding when opening the file
loader = TextLoader(file_path, encoding='utf-8')
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

user_query = input("Enter Your Query...")
retrieved_docs = retriever.get_relevant_documents(user_query)

# Chain 1: Response Chain
response_template = """Your need is to provide responses to the user's shared questions about Trichy from the retrieved documents, showing only relevant answers and presenting them in five bullet points.
% USER RESPONSE
{user_response}

YOUR RESPONSE:
"""
response_prompt_template = PromptTemplate(input_variables=["user_response"], template=response_template)
llm = OpenAI(temperature=1, openai_api_key=openai_api_key)
response_chain = LLMChain(llm=llm, prompt=response_prompt_template)
response_result = response_chain.run(retrieved_docs)

# Chain 2: Locations
location__template = """Given the locations or addresses based on the user's input from user_response and show me in bullet points.
% LOCATION
{user_location}

YOUR RESPONSE:
"""
location_prompt_template = PromptTemplate(input_variables=["user_location"], template=location__template)
location_chain = LLMChain(llm=llm, prompt=location_prompt_template)
location_result = location_chain.run(retrieved_docs)

# Chain 3: explore
explore_template = """Given the other nearby stores or sightseeing spots for explore from retrieved_docs and show me in bullet points.
% SHOPS
{near_places}

YOUR RESPONSE:
"""
near_place_prompt_template = PromptTemplate(input_variables=["near_places"], template=explore_template)
near_place_chain = LLMChain(llm=llm, prompt=near_place_prompt_template)
near_place_result = near_place_chain.run(retrieved_docs)
# Combined Chain
full_chain = SimpleSequentialChain(chains=[response_chain, location_chain, near_place_chain], verbose=True)
   
# # Print the results
# print(f"User Query: {user_query}")
# print(f"Response Result: {response_result}")
# print(f"Locations Result: {location_result}")
# print(f"Near to Explore : {near_place_result}")
from transformers import pipeline

def summarize_text(text, max_length=350, model_name="facebook/bart-large-cnn"):
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, max_length=max_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summary[0]['summary_text']

# Print the summarized results
print("\nSummarized Results:")
print("\nResponse Summary:", summarize_text(response_result))
print("\nLocations Summary:", summarize_text(location_result))
print("\nNearby Places Summary:", summarize_text(near_place_result))
