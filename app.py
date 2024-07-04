from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

# Load a simple dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def tokenize_function(examples):
    return tokenizer(examples['text'], return_special_tokens_mask=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12
)
model = GPT2LMHeadModel(config)


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train']
)


# Train the model
trainer.train()

model.save_pretrained("./gpt2_model")
tokenizer.save_pretrained("./gpt2_tokenizer")


print("Done")
