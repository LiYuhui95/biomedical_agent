from agent.llm import OllamaBackend


llm = OllamaBackend(
    model="qwen2.5:3b"
)

response = llm.generate(
    "What is a pancreatic beta cell? "
    "Answer in one sentence."
)

print(response)