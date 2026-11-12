from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

dataset = load_dataset("stanfordnlp/imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(tokenize, batched=True)
small_train = tokenized["train"].shuffle(seed=42).select(range(1000))
small_test = tokenized["test"].shuffle(seed=42).select(range(200))

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

args = TrainingArguments(output_dir="results", num_train_epochs=1, per_device_train_batch_size=8)
trainer = Trainer(model=model, args=args, train_dataset=small_train, eval_dataset=small_test)
trainer.train()