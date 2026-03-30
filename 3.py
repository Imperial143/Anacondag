import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

texts = [
    "I love this movie", "This is wonderful",
    "Absolutely fantastic film", "I really enjoyed this movie",
    "Brilliant acting and story", "Amazing experience",
    "Superb direction", "I liked this a lot",
    "Great movie overall", "One of the best films",

    "I hate this movie", "This is terrible",
    "Absolutely boring film", "I really disliked this movie",
    "Awful acting and story", "Horrible experience",
    "Poor direction", "I did not like this at all",
    "Worst movie ever", "Very disappointing film",
]

labels = [1]*10 + [0]*10 

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def predict(model, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = logits.softmax(dim=-1)
    return probs

print("Before Fine-Tuning:")
print("Positive:", predict(model, "I love this movie"))
print("Negative:", predict(model, "This is terrible"))

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"]
)
model = get_peft_model(model, lora_config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model.train()

for epoch in range(5):
    for text, label in zip(texts, labels):
        inputs = tokenizer(text, return_tensors="pt")
        labels_tensor = torch.tensor([label])

        outputs = model(**inputs, labels=labels_tensor)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
model.eval()

print("\nAfter Fine-Tuning:")
print("Positive:", predict(model, "I love this movie"))
print("Negative:", predict(model, "This is terrible"))
