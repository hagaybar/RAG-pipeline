from abc import ABC, abstractmethod
from scripts.api_clients.openai.gptApiClient import APIClient


from abc import ABC, abstractmethod

class PromptBuilder(ABC):
    @abstractmethod
    def build(self, query: str, context: str) -> str:
        """Construct a full prompt based on the query and context."""
        pass



# --- Email-specific implementation ---
class EmailPromptBuilder(PromptBuilder):
    def __init__(self, style: str = "default"):
        self.style = style

    def build(self, query: str, context: str) -> str:
        if self.style == "references":
            return self.build_prompt_references(query, context)
        else:
            return self.build_prompt(query, context)

    def build_prompt(self, query: str, context: str) -> str:
        instructions = (
            "You are an AI assistant helping answer questions based on internal email conversations.\n"
            "Use only the information provided in the context below. If the context contains relevant information, "
            "summarize it clearly and concisely. Even if only partial information is available, do your best to synthesize it.\n"
            "If the answer is not found at all, reply with \"I don't know.\"\n"
        )

        return f"""{instructions}
---
{context}
---
Question: {query}
Answer:"""
    
    def build_prompt_references(self, query: str, context: str) -> str:
        instructions = (
            "You are an AI assistant answering questions based on internal email discussions.\n"
            "Each chunk of context is labeled with a number like [1], [2], etc.\n"
            "When answering, cite the relevant chunks using these labels (e.g., [1], [2]).\n"
            "At the end of your answer, do not include a separate list of sources — it will be added automatically.\n"
            "If the context lacks an answer, reply with: \"I don't know.\"\n"
        )

        return f"""{instructions}
---
{context}
---
Question: {query}
Answer:"""