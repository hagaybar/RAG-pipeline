import re
from typing import List, Dict, Tuple

class CitationFormatter:
    def __init__(self, top_chunks: List[Dict]):
        self.top_chunks = top_chunks

    def detect_citations(self, answer_text: str) -> List[int]:
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
        cited_labels = self.detect_citations(answer_text)
        if not cited_labels:
            return answer_text.strip()

        label_mapping = self.build_label_mapping(cited_labels)
        rewritten_answer = self.replace_citations_in_text(answer_text, label_mapping)
        sources_section = self.format_sources(label_mapping)

        return rewritten_answer.strip() + "\n\n" + sources_section
