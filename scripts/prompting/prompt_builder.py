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
    def build(self, query: str, context_chunks: list[dict]) -> str:
        """
        Construct a full prompt based on the query and a list of context chunk dictionaries.
        Each dictionary in context_chunks is expected to contain the text of the chunk
        and potentially other metadata.
        """
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


# --- TextFile-specific implementation ---
class TextFilePromptBuilder(PromptBuilder):
    """
    Builds prompts specifically for question-answering over generic text file content.

    This class implements the `PromptBuilder` interface to construct prompts
    tailored for scenarios where answers are derived from processed text files.
    It supports different prompt "styles" upon initialization (e.g., "default",
    "references") to vary the instructions given to the language model.

    - The "default" style (`_build_prompt` method) instructs the AI to use the
      provided text context to answer the user's query concisely, or to state
      "I don't know" if the answer isn't found.
    - The "references" style (`_build_prompt_references` method) provides more
      detailed instructions, requiring the AI to cite every factual claim in its
      answer using numerical labels (e.g., `[1]`, `[2]`) that correspond to the
      provided context chunks. It also explicitly instructs the AI to omit a
      "Sources" section, as this is typically appended in a later stage by a
      different component (e.g., `CitationFormatter`).
    """
    def __init__(self, style: str = "default"):
        """
        Initializes the TextFilePromptBuilder.

        Args:
            style (str, optional): The style of prompt to generate.
                                   Defaults to "default". Can be "references"
                                   to enable citation-style prompts.
        """
        self.style = style

    def build(self, query: str, context_chunks: list[dict]) -> str:
        """
        Constructs and returns a complete prompt string.

        This method formats each chunk from context_chunks, which are dictionaries
        containing detailed metadata (especially for MARCXML), and then dispatches
        to a style-specific build method.

        Args:
            query (str): The user's query.
            context_chunks (list[dict]): A list of dictionaries, where each
                                         dictionary holds detailed information 
                                         about one retrieved chunk/record,
                                         including its text and metadata.
        Returns:
            str: The fully formatted prompt string.
        """
        
        formatted_context_parts = []
        for i, chunk_dict in enumerate(context_chunks):
            label = f"[{i + 1}]" # Consistent with ChunkRetriever's label_for_llm_context
            
            # Start building the context string for this chunk
            chunk_text_parts = [f"Context Source {label}:"]

            # Prioritize MARCXML-specific structured metadata if available
            record_id = chunk_dict.get('record_id', 'N/A')
            chunk_text_parts.append(f"  Record ID: {record_id}")

            # Safely parse key_metadata_fields_json
            key_metadata_str = chunk_dict.get('key_metadata_fields_json')
            key_metadata = {}
            if key_metadata_str and isinstance(key_metadata_str, str):
                try:
                    key_metadata = json.loads(key_metadata_str)
                except json.JSONDecodeError:
                    # Add a log or handle error if JSON is malformed
                    key_metadata = {'error': 'Could not parse key_metadata_fields_json'}
            
            # Extract specific fields from parsed key_metadata
            title = key_metadata.get('title_main', 'N/A') 
            # Assuming MARCXMLProcessor key_metadata_structure maps 245a to 'title_main'
            # For creator, need to check multiple possible fields based on MARC structure
            creator = key_metadata.get('author_personal_primary', 
                                     key_metadata.get('author_corporate_primary', 
                                                      key_metadata.get('author_meeting_primary', 'N/A')))
            
            chunk_text_parts.append(f"  Title: {title}")
            chunk_text_parts.append(f"  Creator: {creator}")

            pub_year = chunk_dict.get('publication_year_008', 'N/A')
            chunk_text_parts.append(f"  Publication Year: {pub_year}")

            languages = chunk_dict.get('languages', []) # Should be a list directly
            if languages:
                 chunk_text_parts.append(f"  Languages: {', '.join(languages)}")
            
            # Add the main text of the chunk
            main_chunk_content = chunk_dict.get('text', '') # 'text' key from ChunkRetriever
            chunk_text_parts.append(f"  Content: {main_chunk_content}")
            
            formatted_context_parts.append("\n".join(chunk_text_parts))

        final_context_str = "\n---\n".join(formatted_context_parts)

        if self.style == "references":
            return self._build_prompt_references(query, final_context_str)
        return self._build_prompt(query, final_context_str)

    def _build_prompt(self, query: str, formatted_context: str) -> str:
        """
        Constructs a standard prompt for question-answering based on text file context.

        The prompt instructs the AI to use only the provided context, summarize
        information clearly and concisely, synthesize partial information if
        necessary, and reply with "I don't know" if the answer cannot be found
        in the context.

        Args:
            query (str): The user's query.
            context (str): The contextual text data to be used for answering.

        Returns:
            str: A formatted prompt string for the "default" question-answering style.
        """
        instructions = (
            "You are an AI assistant helping answer questions based on information from various documents.\n"
            "The context below provides information extracted from these documents. Each 'Context Source' is labeled (e.g., [1], [2]) "
            "and may include specific metadata like Record ID, Title, Creator, Publication Year, and Languages, followed by its main Content.\n"
            "Use only the information provided in the context. If the context contains relevant information, "
            "summarize it clearly and concisely. Even if only partial information is available, do your best to synthesize it.\n"
            "When referring to a specific source, you can use its label (e.g., 'According to source [1]...').\n"
            "If the answer is not found at all, reply with \"I don't know.\"\n"
        )

        return f"""{instructions}
---
Context from Documents:
{formatted_context}
---
Question: {query}
Answer:"""

    def _build_prompt_references(self, query: str, formatted_context: str) -> str:
        """
        Constructs a prompt specifically designed to make the AI cite sources
        from the provided text file context.

        The prompt instructs the AI that the context chunks are labeled (e.g., `[1]`,
        `[2]`), that every factual claim in its answer must be cited using these
        labels, and that it should not include a "Sources" section (as this will
        be handled by a separate formatting step). If no answer can be derived,
        it should reply "I don't know."

        Args:
            query (str): The user's query.
            context (str): The contextual text data, expected to contain
                           reference labels (e.g., "[1] Some text...").

        Returns:
            str: A formatted prompt string for the "references" question-answering style.
        """
        instructions = (
            "You are an AI assistant answering questions based on information from various documents.\n"
            "The context below provides information extracted from these documents. Each 'Context Source' is labeled (e.g., [1], [2]) "
            "and includes specific metadata like Record ID, Title, Creator, Publication Year, and Languages, followed by its main Content.\n\n"
            "✅ When writing your answer:\n"
            "- **Every factual claim must be cited** using the source labels (e.g., [1], [2]).\n"
            "- These citations help the user verify where the information came from.\n"
            "- If you don't include any [number] citations, your answer will be considered incomplete.\n\n"
            "❗ Do not include a 'Sources' section. It will be added automatically later.\n"
            "If no answer can be derived from the context, reply: 'I don't know.'\n"
        )

        return f"""{instructions}
---
Context from Documents:
{formatted_context}
---
Question: {query}
Answer:"""
    
    # This method seems to be a duplicate of EmailPromptBuilder's build_prompt_references.
    # It should be specific to TextFilePromptBuilder if kept, or removed if covered by the above.
    # For now, I'll assume the _build_prompt_references is the one to be used by TextFilePromptBuilder.
    # def build_prompt_references(self, query: str, context: str) -> str:
    #     """
    #     Constructs a prompt specifically designed to make the AI cite sources
    #     from the provided context.
    #     ...
    #     """
    #     instructions = (
    #         "You are an AI assistant answering questions based on internal email discussions.\n" # This seems like a copy-paste from EmailPromptBuilder
    #         "Each context chunk is labeled [1], [2], etc.\n\n"
    #         "✅ When writing your answer:\n"
    #         "- **Every factual claim must be cited** using these labels (e.g., [1], [2]).\n"
    #         "- These citations help the user verify where the information came from.\n"
    #         "- If you don't include any [number] citations, your answer will be considered incomplete.\n\n"
    #         "❗ Do not include a 'Sources' section. It will be added automatically later.\n"
    #         "If no answer can be derived from the context, reply: 'I don't know.'\n"
    #     )

    #     return f"""{instructions}
# ---
# {context}
# ---
# Question: {query}
# Answer:"""