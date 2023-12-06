from flask import Flask, render_template, request
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os

app = Flask(__name__)

openai_api_key = "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"
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load documents
file_path = os.path.join(script_dir, 'Trichy.txt')
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

# Prompt template for the response
response_template = """Your need is to provide responses to the user's shared questions about Trichy from the retrieved documents, showing without addresses and presenting them in five bullet points.
% USER RESPONSE
{user_response}

YOUR RESPONSE:
"""
response_prompt_template = PromptTemplate(input_variables=["user_response"], template=response_template)
llm = OpenAI(temperature=1, openai_api_key=openai_api_key)
response_chain = LLMChain(llm=llm, prompt=response_prompt_template)

# Prompt template for locations
location_template = """Given the locations or addresses based on the user's input from user_response.
% LOCATION
{user_location}

YOUR RESPONSE:
"""
location_prompt_template = PromptTemplate(input_variables=["user_location"], template=location_template)
location_chain = LLMChain(llm=llm, prompt=location_prompt_template)

# Prompt template for exploring
explore_template = """Given the other nearby stores or sightseeing spots for explore from retrieved_docs.
% SHOPS
{near_places}

YOUR RESPONSE:
"""
near_place_prompt_template = PromptTemplate(input_variables=["near_places"], template=explore_template)
near_place_chain = LLMChain(llm=llm, prompt=near_place_prompt_template)

# Combined Chain
full_chain = SimpleSequentialChain(chains=[response_chain, location_chain, near_place_chain], verbose=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    user_query = ""
    response_result = ""
    location_result = []
    near_place_result = ""

    if request.method == 'POST':
        user_query = request.form['user_query']
        retrieved_docs = retriever.get_relevant_documents(user_query)

        # Run the chains with the retrieved documents
        response_result = response_chain.run(retrieved_docs)
        location_result = location_chain.run(retrieved_docs)
        near_place_result = near_place_chain.run(retrieved_docs)

    return render_template('result.html', user_query=user_query,
                           response_result=response_result,
                           location_result=location_result,
                           near_place_result=near_place_result)


if __name__ == '__main__':
    app.run(debug=True)
