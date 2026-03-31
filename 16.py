from transformers import pipeline
model_name = "gpt2"
generator = pipeline("text-generation", model=model_name,device =-1)
def generate_article(prompt):
    result = generator(prompt, max_length=250, num_return_sequences=1)
    return result[0]['generated_text']
engineered_prompt = """
Write a short and clear article on the topic: Applications of Generative AI in Healthcare.
Structure:
1. Introduction
2. Applications (diagnosis, drug discovery, medical imaging)
3. Benefits
4. Conclusion
Use simple language.
"""
article = generate_article(engineered_prompt)
print("Prompt Used:\n", engineered_prompt)
print("\nGenerated Article:\n")
print(article)
