from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt",truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

question = "If 1 pen costs 10 rupees, what is the cost of 5 pens?"

zero_prompt = f"""
Answer the question:

{question}
"""

zero_output = generate_text(zero_prompt)

few_prompt = f"""
Solve the following:

Q: If 1 pen costs 10 rupees, what is the cost of 3 pens?
A: 3 * 10 = 30 rupees

Q: If 1 pencil costs 10 rupees, what is the cost of 4 pencils?
A: 4 * 10 = 40 rupees

Q: If 1 book costs 10 rupees , what is the cost of 5 books?
A: 5 * 10 = 50 rupees

Q: {question}
A:
"""

few_output = generate_text(few_prompt)


cot_prompt = f"""
Let's slove step by step and then give final answer:

Q: {question}
Final Answer:
"""

cot_output = generate_text(cot_prompt)


print("===== ZERO-SHOT OUTPUT =====")
print(zero_output)

print("\n===== FEW-SHOT OUTPUT =====")
print(few_output)

print("\n===== CHAIN-OF-THOUGHT OUTPUT =====")
print(cot_output)
