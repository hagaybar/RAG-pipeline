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
            "Each context chunk is labeled [1], [2], etc.\n\n"
            "✅ When writing your answer:\n"
            "- **Every factual claim must be cited** using these labels (e.g., [1], [2]).\n"
            "- These citations help the user verify where the information came from.\n"
            "- If you don't include any [number] citations, your answer will be considered incomplete.\n\n"
            "❗ Do not include a 'Sources' section. It will be added automatically later.\n"
            "If no answer can be derived from the context, reply: 'I don't know.'\n"
        )

        return f"""{instructions}
---
{context}
---
Question: {query}
Answer:"""