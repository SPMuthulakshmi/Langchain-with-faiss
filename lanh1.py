from langchain.chains import LLMChain
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

# Get the absolute path to the directory of the script
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

# Prompt template for dish
template = """Your need is to come up with top Colleges in the location display in only five bullet points.
% USER LOCATION
{user_location}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_location"], template=template)

# Language model chain
llm = OpenAI(temperature=1, openai_api_key=openai_api_key)
dish_chain = LLMChain(llm=llm, prompt=prompt_template)

# Combine retriever and dish chain
def combined_chain(query):
    relevant_docs = retriever.get_relevant_documents(query)
    response = dish_chain.run(query)
    return relevant_docs, response

# Run the combined chain
user_query = "Trichy"
result = combined_chain(user_query)
print(result)
