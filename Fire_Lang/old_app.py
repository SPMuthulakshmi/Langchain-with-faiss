from flask import Flask, render_template, request, jsonify
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from getpass import getpass
import os

app = Flask(__name__)

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

# Prompt template for the best location
response_template = """Your need is to come up with the shop names for the user shared in Trichy and show me in five bullet points only.
% RESPONSE
{user_response}

YOUR RESPONSE:
"""
response_prompt_template = PromptTemplate(input_variables=["user_response"], template=response_template)
llm = OpenAI(temperature=1, openai_api_key=openai_api_key)
# Chain 1: Location Chain
response_chain = LLMChain(llm=llm, prompt=response_prompt_template)

# Chain 2: Recipe
location_template = """Given the locations, show me the reviews and price range show me in INR.
% DISH
{user_location}

YOUR RESPONSE:
"""
response_prompt_template = PromptTemplate(input_variables=["user_location"], template=location_template)
location_chain = LLMChain(llm=llm, prompt=response_prompt_template)

# Chain 3: Ingredient Cost
explore_template = """Given the nearby shop the user shared, show me if the shops are for men, women, or kids.
% near_places
{near_places}

YOUR RESPONSE:
"""
near_place_prompt_template = PromptTemplate(input_variables=["near_places"], template=explore_template)
near_place_chain = LLMChain(llm=llm, prompt=near_place_prompt_template)

# Combined Chain
full_chain = SimpleSequentialChain(chains=[response_chain, location_chain, near_place_chain], verbose=True)
@app.route('/', methods=['GET','POST'])  # This is the root path
def index():
    # You can render the main page template here if needed
    return render_template('index.html')

@app.route('/process_query', methods=['GET'])
def process_query():
    try:
        # Get user input for the query from the request
        user_query = request.args.get('query')

        # Retrieve relevant documents from the FAISS index
        retrieved_docs = retriever.get_relevant_documents(user_query)

        # Run the chains with the retrieved documents
        response_result = response_chain.run(retrieved_docs)
        location_result = location_chain.run(retrieved_docs)
        near_place_result = near_place_chain.run(retrieved_docs)

        # Return the results in the response
        return render_template('index.html', user_query=user_query,
                               response_result=response_result,
                               location_result=location_result,
                               near_place_result=near_place_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
