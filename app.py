from flask import Flask
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig

app = Flask(__name__)

model_name = "gpt2"
model_config = GenerationConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=-1)

@app.route("/")
def generate_text():
    prompt = "What is the name of the moon?"
    generated_text = text_generator(prompt, max_length=200, do_sample=True, temperature=0.7)[0]['generated_text']
    return generated_text

if __name__ == '__main__':
    app.run()













