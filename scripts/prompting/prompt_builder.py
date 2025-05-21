"""
This module contains classes for dynamically constructing prompts for language
models. It includes the `PromptBuilder` abstract base class for defining a
prompt construction interface, and a specific implementation `EmailPromptBuilder`
which creates prompts for answering questions based on email context, supporting
styles like in-text citations.
"""
from abc import ABC, abstractmethod
from scripts.api_clients.openai.gptApiClient import APIClient


from abc import ABC, abstractmethod

class PromptBuilder(ABC):
    """
    Abstract base class for prompt construction.

    This class defines the standard interface for building prompts. Concrete
    subclasses must implement the `build` method, which is responsible for
    combining a user's query and relevant context into a complete prompt
    string to be sent to a language model.
    """
    @abstractmethod
    def build(self, query: str, context: str) -> str:
        """Construct a full prompt based on the query and context."""
        pass



# --- Email-specific implementation ---
class EmailPromptBuilder(PromptBuilder):
    """
    Builds prompts specifically for question-answering over email context.

    This class implements the `PromptBuilder` interface to construct prompts
    tailored for scenarios where answers are derived from email conversations.
    It supports different prompt "styles" upon initialization (e.g., "default",
    "references") to vary the instructions given to the language model.

    - The "default" style (`build_prompt` method) instructs the AI to use the
      provided email context to answer the user's query concisely, or to state
      "I don't know" if the answer isn't found.
    - The "references" style (`build_prompt_references` method) provides more
      detailed instructions, requiring the AI to cite every factual claim in its
      answer using numerical labels (e.g., `[1]`, `[2]`) that correspond to the
      provided context chunks. It also explicitly instructs the AI to omit a
      "Sources" section, as this is typically appended in a later stage by a
      different component (e.g., `CitationFormatter`).
    """
    def __init__(self, style: str = "default"):
        """
        Initializes the EmailPromptBuilder.

        Args:
            style (str, optional): The style of prompt to generate.
                                   Defaults to "default". Can be "references"
                                   to enable citation-style prompts.
        """
        self.style = style

    def build(self, query: str, context: str) -> str:
        """
        Constructs and returns a complete prompt string.

        This method dispatches to a style-specific build method
        (`build_prompt` or `build_prompt_references`) based on the
        `self.style` attribute set during initialization.

        Args:
            query (str): The user's query.
            context (str): The contextual information (e.g., retrieved email
                           chunks) to be included in the prompt.

        Returns:
            str: The fully formatted prompt string.
        """
        if self.style == "references":
            return self.build_prompt_references(query, context)
        else:
            return self.build_prompt(query, context)

    def build_prompt(self, query: str, context: str) -> str:
        """
        Constructs a standard prompt for question-answering based on email context.

        The prompt instructs the AI to use only the provided context, summarize
        information clearly and concisely, synthesize partial information if
        necessary, and reply with "I don't know" if the answer cannot be found
        in the context.

        Args:
            query (str): The user's query.
            context (str): The contextual email data to be used for answering.

        Returns:
            str: A formatted prompt string for the "default" question-answering style.
        """
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
        """
        Constructs a prompt specifically designed to make the AI cite sources
        from the provided context.

        The prompt instructs the AI that the context chunks are labeled (e.g., `[1]`,
        `[2]`), that every factual claim in its answer must be cited using these
        labels, and that it should not include a "Sources" section (as this will
        be handled by a separate formatting step). If no answer can be derived,
        it should reply "I don't know."

        Args:
            query (str): The user's query.
            context (str): The contextual email data, expected to contain
                           reference labels (e.g., "[1] Some text...").

        Returns:
            str: A formatted prompt string for the "references" question-answering style.
        """
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