# scripts/formatting/format_sources.py

from typing import List, Dict
import pandas as pd

def format_sources(top_chunks: List[Dict], label_field: str = "rank", fields: List[str] = ["Sender", "Received", "Subject"]) -> str:
    """
    Formats a deduplicated, date-sorted 'Sources' section from metadata.

    Args:
        top_chunks (List[Dict]): List of retrieved chunks with metadata.
        label_field (str): Field used for numbering (default: "rank").
        fields (List[str]): Metadata fields to include per source line.

    Returns:
        str: A formatted Sources section.
    """
    if not top_chunks:
        return ""

    seen_senders = set()
    sources_entries = []

    for chunk in top_chunks:
        metadata = chunk.get("metadata", {})
        sender = metadata.get("Sender", "Unknown Sender")

        if sender not in seen_senders:
            received_raw = metadata.get("Received", "Unknown Date")
            try:
                received = pd.to_datetime(received_raw).strftime("%Y-%m-%d")
            except Exception:
                received = "Unknown Date"
            subject = metadata.get("Subject", "No Subject")
            label = chunk.get(label_field, "N/A")

            sources_entries.append({
                "label": label,
                "sender": sender,
                "received": received,
                "subject": subject
            })
            seen_senders.add(sender)

    # Sort entries by received date
    sources_entries.sort(key=lambda x: x["received"])

    sources_lines = ["Sources:"]
    for entry in sources_entries:
        sources_lines.append(f"[{entry['label']}] {entry['sender']}, {entry['received']} – {entry['subject']}")

    return "\n".join(sources_lines)
