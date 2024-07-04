from transformers import pipeline
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import GPT2Tokenizer

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./gpt2_model")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_tokenizer")

# Create a text generation pipeline
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Generate text
output = text_generator("Once upon a time", max_length=50)
print(output)