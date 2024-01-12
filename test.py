# # Use a pipeline as a high-level helper
# from transformers import pipeline

# # pipe = pipeline("text-generation", model="bigscience/bloom") too large to use
# pipe = pipeline("text-generation", model="bigscience/bloom-3b")

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-3b")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3b")

print('check')

prompt = "It was a dark and stormy night"
result_length = 50
inputs = tokenizer(prompt, return_tensors="pt")

# Greedy Search
print(tokenizer.decode(model.generate(inputs["input_ids"], 
                       max_length=result_length
                      )[0]))