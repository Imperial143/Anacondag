from transformers import pipeline
generator = pipeline("text-generation", model="gpt2", device=-1)
print("Enter incomplete prompt (type 'exit' to quit):\n")
while True:
    prompt = input("Input: ")
    
    if prompt.lower() == "exit":
        print("Bye!")
        break
    result = generator(prompt, max_length=50)
    print("\nCompleted Text:\n", result[0]['generated_text'])
    print("\n" + "-"*40 + "\n")
