from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from getpass import getpass
import os

# Get OpenAI API key
openai_api_key = getpass("Enter your OpenAI API key: ")
print("API key successfully received.")

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

# Chain 1: Location Chain
location_template = """Your need is to come up with the shop names for the user shared in Trichy and show me in only five bullet points only.
% USER LOCATION
{user_location}

YOUR RESPONSE:
"""
location_prompt_template = PromptTemplate(input_variables=["user_location"], template=location_template)
llm = OpenAI(temperature=1, openai_api_key=openai_api_key)
location_chain = LLMChain(llm=llm, prompt=location_prompt_template)

# Chain 2: Recipe
location__template = """Given the locations, show me the reviews and price range show me in INR.
% DISH
{user_location}

YOUR RESPONSE:
"""
location_prompt_template = PromptTemplate(input_variables=["user_location"], template=location__template)
location_chain = LLMChain(llm=llm, prompt=location_prompt_template)

# Chain 3: Ingredient Cost
explore_template = """Given the near by shop the user shared, show me the shops are mens or womens or kids.
% near_places
{near_places}

YOUR RESPONSE:
"""
near_place_prompt_template = PromptTemplate(input_variables=["near_places"], template=explore_template)
near_place_chain = LLMChain(llm=llm, prompt=near_place_prompt_template)

# Combined Chain
full_chain = SimpleSequentialChain(chains=[location_chain, location_chain, near_place_chain], verbose=True)

def get_chains_results(user_query):
    # Retrieve relevant documents from the FAISS index
    retrieved_docs = retriever.get_relevant_documents(user_query)

    # Run each chain with the retrieved documents
    response_result = location_chain.run(retrieved_docs)
    location_result = location_chain.run(retrieved_docs)
    near_place_result = near_place_chain.run(retrieved_docs)

    return response_result, location_result, near_place_result

# Take user input for the query
user_query = input("Enter your query: ")
response_result, location_result, near_place_result = get_chains_results(user_query)

# Print the results
print(f"User Query: {user_query}")
print(f"Location Result: {response_result}")
print(f"Recipe Result: {location_result}")
print(f"Ingredient Cost Result: {near_place_result}")
