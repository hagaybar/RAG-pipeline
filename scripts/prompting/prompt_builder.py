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