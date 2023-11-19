from flask import Flask, render_template, request, jsonify
import os
import pickle
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
import requests
import torch
from transformers import AutoModelForCausalLM,AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import PeftConfig
from peft import PeftModel


# Creating the flask application over here
app = Flask(__name__)
import os
os.environ['TRANSFORMERS_CACHE'] = 'D:/DataMentor/model'

bnb_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,

    # nf4 is the normal 4 bit format that is again provided by LoRa
    bnb_4bit_quant_type="nf4",

    # over here we are computing 16 bit precision
    bnb_4bit_compute_dtype=torch.bfloat16,
)

PEFT_MODEL="MANMEET75/Mistral-7B-v0.1-fine-tuned"

config=PeftConfig.from_pretrained(PEFT_MODEL)
model=AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,

)

tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model=PeftModel.from_pretrained(model,PEFT_MODEL)


generation_config=model.generation_config
generation_config.max_new_tokens=70
generation_config.temperature = 0.2
generation_config.top_p = 0.2
generation_config.num_return_sequences=1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

DEVICE = "cuda:0"
def generate_response(question: str) -> str:
    prompt = f"""
<human>{question}
<assistant>
    """.strip()

    encoding = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    assistant_start = "<assistant>"
    response_start = response.find(assistant_start)

    # Extracting text after the <assistant> tag
    assistant_text = response[response_start + len(assistant_start):].strip()

    # Find the index of the last dot in the extracted text
    last_dot_index = assistant_text.rfind('.')

    # Extract the text until the ending of the last dot
    if last_dot_index != -1:
        assistant_text = assistant_text[:last_dot_index + 1].strip()

    # Removing double quotes and the colon from the extracted text
    assistant_text = assistant_text.replace('"', '').replace(':', '')

    return assistant_text



@app.route('/static/<path:filename>')
def serve_static(filename):
    return app.send_static_file(filename)


@app.route("/")
def home():
    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()  # Use request.get_json() to parse JSON data
    if data and 'input' in data:
        input_question = data['input']
        response = generate_response(input_question)
        return jsonify({'response': response})
    else:
        return jsonify({'error': 'Invalid input format'}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 8080)