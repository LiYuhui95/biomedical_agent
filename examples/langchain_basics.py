from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


model = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)

messages = [
    (
        "system",
        "You are a concise biomedical assistant.",
    ),
    (
        "human",
        "What is TNIK?",
    ),
]

response = model.invoke(
    messages
)

print("Response object:")
print(type(response))

print("\nContent:")
print(response.content)

print("\nMetadata:")
print(response.response_metadata)


class BiomedicalTerms(BaseModel):
    terms: list[str] = Field(
        description=(
            "Real biomedical entities and "
            "concepts found in the question"
        )
    )


structured_model = (
    model.with_structured_output(
        BiomedicalTerms
    )
)

result = structured_model.invoke(
    [
        (
            "system",
            "Extract biomedical search terms. "
            "Do not include generic wording.",
        ),
        (
            "human",
            (
                "Is TNIK a promising therapeutic "
                "target for pulmonary fibrosis?"
            ),
        ),
    ]
)

print("\nStructured result:")
print(result)

print("\nResult type:")
print(type(result))

print("\nTerms:")
print(result.terms)