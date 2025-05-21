"""
This module contains the `CitationFormatter` class, designed to process
AI-generated text containing citation markers (e.g., `[1]`). It detects these
markers, renumbers them sequentially, and appends a formatted list of sources
based on the metadata of the originally cited text chunks, such as sender,
date, and subject for email documents.
"""
import re
from typing import List, Dict, Tuple

class CitationFormatter:
    """
    Handles the post-processing of AI-generated answers to correctly format
    in-text citations and append a corresponding list of sources.

    When an answer from a language model includes citation markers (e.g., `[1]`, `[2]`),
    this class ensures these citations are accurate and properly presented.
    It is initialized with `top_chunks`, which is a list of dictionaries, each
    representing a source document/chunk that the AI might have cited. Each chunk's
    dictionary should contain its metadata and an original `rank` that the AI might
    have used for its initial citation.

    The process involves:
    1.  **Detecting Citations**: The `detect_citations` method finds all numeric
        citation markers (like `[1]`) within the raw answer text.
    2.  **Renumbering and Rewriting**: If citations are found, `build_label_mapping`
        creates a map from the original citation numbers in the text to new,
        sequential, 1-based numbers. `replace_citations_in_text` then updates the
        answer text with these new citation numbers. This ensures citations are
        dense and correctly ordered, especially if the AI's original citations
        were sparse or out of order.
    3.  **Formatting Sources**: The `format_sources` method generates a "Sources:"
        section. It iterates through the remapped citations that actually appear
        in the answer. For each, it retrieves the corresponding original chunk's
        metadata (e.g., Sender, Received date, Subject for emails) from `top_chunks`
        (using the original `rank`) and formats it into a readable source line,
        prefixed with the new citation number.
    4.  **Finalizing Answer**: The `finalize_answer` method orchestrates these steps,
        returning the answer text with updated citations followed by the generated
        sources section. If no citations are detected, the original answer is returned.
    """
    def __init__(self, top_chunks: List[Dict]):
        """
        Initializes the CitationFormatter.

        Args:
            top_chunks (List[Dict]): A list of dictionaries, where each dictionary
                                     represents a source chunk retrieved by the RAG
                                     system. Each chunk dictionary is expected to
                                     contain a 'rank' key (the original citation
                                     number potentially used by the LLM) and a
                                     'metadata' dictionary with details like
                                     'Sender', 'Received', 'Subject' for formatting
                                     the sources list.
        """
        self.top_chunks = top_chunks

    def detect_citations(self, answer_text: str) -> List[int]:
        """
        Finds all unique numerical citation markers (e.g., `[1]`, `[23]`)
        within the provided answer text.

        Args:
            answer_text (str): The text (typically an AI-generated answer) to
                               scan for citation markers.

        Returns:
            List[int]: A list of unique integers representing the citation
                       numbers found in the text, in the order of their first
                       appearance.
        """
        citations = re.findall(r'\[(\d+)\]', answer_text)
        unique_citations = []
        for c in citations:
            num = int(c)
            if num not in unique_citations:
                unique_citations.append(num)
        return unique_citations

    def build_label_mapping(self, cited_labels: List[int]) -> Dict[int, int]:
        """
        Builds a mapping from old citation numbers to new 1-based numbering.
        """
        return {old: new for new, old in enumerate(cited_labels, start=1)}

    def replace_citations_in_text(self, text: str, label_mapping: Dict[int, int]) -> str:
        """
        Replaces old citation numbers with new ones in the answer text.
        """
        def repl(match):
            old_label = int(match.group(1))
            new_label = label_mapping.get(old_label, old_label)
            return f"[{new_label}]"

        return re.sub(r'\[(\d+)\]', repl, text)

    def format_sources(self, label_mapping: Dict[int, int]) -> str:
        """
        Creates the Sources section based on remapped citation numbers.
        """
        if not label_mapping or not self.top_chunks:
            return ""

        label_to_chunk = {chunk["rank"]: chunk for chunk in self.top_chunks}
        sources_lines = ["Sources:"]

        for old_label, new_label in label_mapping.items():
            chunk = label_to_chunk.get(old_label)
            if chunk:
                metadata = chunk.get("metadata", {})
                sender = metadata.get("Sender", "Unknown Sender")
                received = metadata.get("Received", "Unknown Date")
                subject = metadata.get("Subject", "No Subject")
                sources_lines.append(f"[{new_label}] {sender}, {received} – {subject}")

        return "\n".join(sources_lines)

    def finalize_answer(self, answer_text: str) -> str:
        """
        Orchestrates the full citation processing workflow for an answer text.

        This method performs the following steps:
        1. Detects citation markers (e.g., `[1]`) in the input `answer_text`
           using `detect_citations`.
        2. If citations are found:
           a. Creates a mapping from original citation numbers to new, sequential,
              1-based numbers using `build_label_mapping`.
           b. Updates the `answer_text` by replacing original citation markers
              with the new, renumbered ones using `replace_citations_in_text`.
           c. Generates a formatted "Sources:" section string using `format_sources`,
              based on the renumbered citations and the `top_chunks` provided
              during initialization.
           d. Appends the "Sources:" section to the renumbered answer text.
        3. If no citations are found in the original `answer_text`, the method
           returns the original answer text, stripped of leading/trailing whitespace.

        Args:
            answer_text (str): The raw answer text from the language model,
                               potentially containing citation markers.

        Returns:
            str: The processed answer string with renumbered in-text citations
                 and an appended, formatted list of sources, or the original
                 stripped answer if no citations were present.
        """
        cited_labels = self.detect_citations(answer_text)
        if not cited_labels:
            return answer_text.strip()

        label_mapping = self.build_label_mapping(cited_labels)
        rewritten_answer = self.replace_citations_in_text(answer_text, label_mapping)
        sources_section = self.format_sources(label_mapping)

        return rewritten_answer.strip() + "\n\n" + sources_section
