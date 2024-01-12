from transformers import (BloomTokenizerFast,
                          BloomForTokenClassification,
                          DataCollatorForTokenClassification, 
                          AutoModelForTokenClassification, 
                          TrainingArguments, Trainer)
from datasets import load_dataset
import torch
import os
torch.cuda.empty_cache() 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"

model_name = "bloom-560m"
tokenizer = BloomTokenizerFast.from_pretrained(f"bigscience/{model_name}", add_prefix_space=True)
model = BloomForTokenClassification.from_pretrained(f"bigscience/{model_name}")

# datasets = load_dataset('conll2003')
datasets = load_dataset('csv', data_files='all_emails.csv')
#print(datasets)
from datasets import DatasetDict

# Splitting the dataset into 80% training and 20% testing
#train_test_split = datasets['train'].train_test_split(test_size=0.2, seed=42)
datasets = datasets['train'].train_test_split(test_size=0.2, seed=42)
#print(train_test_split)
# The resulting splits are contained in a DatasetDict object
# splits = DatasetDict({
#     'train': train_test_split['train'],
#     'test': train_test_split['test']
# })

# Now you have splits['train'] and splits['test'] as your training and testing sets


print("Dataset Object Type:", type(datasets["train"]))
print("Training Examples:", len(datasets["train"]))

datasets["train"][100]

# print("Dataset Object Type:", type(datasets["train"]))
# print("Training Examples:", len(datasets["train"]))

# datasets["train"][100]

example = datasets["train"][0]
print(example)
tokenized_input = tokenizer(example["email"])
# tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
print(tokenized_input)

# tonkenize all input
def tokenizeInputs(inputs):
    
#     tokenized_inputs = tokenizer(inputs["email"], max_length = 512, truncation=True, is_split_into_words=True)
#     word_ids = tokenized_inputs.word_ids()
#     ner_tags = inputs["ner_tags"]
#     labels = [ner_tags[word_id] for word_id in word_ids]
#     tokenized_inputs["labels"] = labels
    
    tokenized_inputs = tokenizer(inputs["email"])
    
    word_ids = tokenized_inputs.word_ids()
    label = inputs["label"]
    labels = label
    tokenized_inputs["labels"] = [labels]
    
    return tokenized_inputs

example = datasets["train"][100]
#print(datasets["train"][100]["label"])
tokenizeInputs(example)
print(datasets)
tokenized_datasets = datasets.map(tokenizeInputs)

token_count = 0
for sample in tokenized_datasets["train"]:
    token_count = token_count + 1


print("Tokens in Training Set:", token_count)

tokenized_datasets = tokenized_datasets.remove_columns(["email","label"])
print(tokenized_datasets)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

model = AutoModelForTokenClassification.from_pretrained(f"bigscience/{model_name}", num_labels=12).cuda()

print("Parameters:", model.num_parameters())
print("Expected Input Dict:", model.main_input_name )

# Estimate FLOPS needed for one training example
sample = tokenized_datasets["train"][0]
sample["input_ids"] = torch.Tensor(sample["input_ids"])
flops_est = model.floating_point_ops(input_dict = sample, exclude_embeddings = False)

print("FLOPS needed per Training Sample:", flops_est )

training_args = TrainingArguments(
    output_dir="./results",
    save_strategy= "epoch", # Disabled for runtime evaluation 
    evaluation_strategy="steps", #"steps", # Disabled for runtime evaluation 
    eval_steps = 500,
    learning_rate=2e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=2,
    weight_decay=0.01,
    report_to="none",
    #fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)



trainer.train()

eval_results = trainer.evaluate()
print(f"Eval Loss: {eval_results['eval_loss']}")