from flask import Flask, request, jsonify, render_template
from flask_restful import Api, Resource
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os

app = Flask(__name__)
api = Api(app)

class LanguageModelAPI(Resource):
    def __init__(self):
        self.openai_api_key = "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"
        script_dir = os.path.dirname(os.path.abspath(__file__))

        file_path = os.path.join(script_dir, 'Trichy_vec.txt')
        loader = TextLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

        db = FAISS.from_documents(texts, embeddings)
        self.retriever = db.as_retriever()

        # User response
        response_template = """Your need is to come up with the response for the user shared and show me in five bullet points only.
        % RESPONSE
        {user_response}

        YOUR RESPONSE:
        """
        response_prompt_template = PromptTemplate(input_variables=["user_response"], template=response_template)
        llm = OpenAI(temperature=1, openai_api_key=self.openai_api_key)
        self.response_chain = LLMChain(llm=llm, prompt=response_prompt_template)

        # Chain 2: Recipe
        location_template = """Given the exact locations and show me near some site seeing and small stores to explore.
        % LOCATION
        {user_location}

        YOUR RESPONSE:
        """
        location_prompt_template = PromptTemplate(input_variables=["user_location"], template=location_template)
        self.location_chain = LLMChain(llm=llm, prompt=location_prompt_template)

        # Create combined chain
        self.full_chain = SimpleSequentialChain(chains=[self.response_chain, self.location_chain], verbose=True)

    def get(self):
        return render_template('index.html', message="This is the Language Model API. Send a POST request with 'user_query' to get results.")

    def post(self):
        data = request.get_json()

        user_query = data.get('user_query', '')

        # Run the chains with the retrieved documents
        response_result = self.response_chain.run({'user_response': user_query})

        response_result = self.location_chain.run({'user_location': response_result})

        # Split locations into a list
        locations_list = response_result.split('\n') if response_result else []

        return render_template('index.html', user_query=user_query, response_result=response_result, locations_list=locations_list)

api.add_resource(LanguageModelAPI, '/api/language_model')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
