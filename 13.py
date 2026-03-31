from transformers import pipeline
model_name = "distilbert-base-cased-distilled-squad"
qa_model = pipeline("question-answering", model=model_name)
context = """
Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.
It enables systems to learn from data, make decisions, and solve problems.
AI is widely used in healthcare for diagnosis, in education for personalized learning,
and in industries for automation and efficiency.
"""
print("Ask questions (type 'exit' to stop)\n")
while True:
    question = input("Enter your question: ")
    
    if question.lower() == "exit":
        print("Bye!")
        break
    result = qa_model(question=question, context=context)
    
    print("Answer:", result["answer"])
    print("Confidence:", round(result["score"], 3))
    print()
