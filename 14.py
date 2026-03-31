from transformers import pipeline
summarizer = pipeline("summarization")
print("Enter paragraph (type 'exit' to quit):\n")
while True:
    text = input("Input: ")
    
    if text.lower() == "exit":
        print("Bye!")
        break
    summary = summarizer(text, max_length=80, min_length=30, do_sample=False)
    print("\nOriginal Text:\n", text)
    print("\nSummary:\n", summary[0]['summary_text'])
    print("\n" + "-"*40 + "\n")
