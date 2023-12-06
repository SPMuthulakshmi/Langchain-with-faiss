from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import os
from langchain.llms import Runnable

class HuggingFaceRunnable(Runnable):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run(self, inputs):
        input_text = inputs['text']
        inputs = self.tokenizer(input_text, return_tensors='pt')
        outputs = self.model(**inputs)
        return self.tokenizer.decode(outputs['logits'][0], skip_special_tokens=True)



# Use the HuggingFaceRunnable in your code
model_name = "gpt2"  # Replace with the desired transformer model
huggingface_llm = HuggingFaceRunnable(model_name)

# Assuming you want to use a prompt to interact with the model
user_query = input("Enter your query: ")
response_template = """Your need is to come up with the response for the user shared and show me in five bullet points only.
% RESPONSE
{user_response}

YOUR RESPONSE:
"""

response_prompt_template = PromptTemplate(input_variables=["user_response"], template=response_template)

response_chain = LLMChain(llm=huggingface_llm, prompt=response_prompt_template)

response_result = huggingface_llm.run({'text': user_query})

# Print the results
print(f"User Query: {user_query}")
print(f"Response Result: {response_result}")
