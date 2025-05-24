# Column name for the final text content to be embedded
COL_CHUNK = "Chunk"

# Column name for data from EmailFetcher
COL_CLEANED_BODY = "Cleaned Body"
COL_RAW_BODY = "Raw Body" # (If needed, from EmailFetcher)
COL_SUBJECT = "Subject" # (If needed)
COL_SENDER = "Sender"   # (If needed)
COL_RECEIVED = "Received" # (If needed)
COL_ENTRY_ID = "EntryID" # (If needed)


# Column name for data from TextFileFetcher and XMLFetcher
COL_CLEANED_TEXT = "Cleaned Text"
COL_RAW_TEXT = "Raw Text" # (If needed)
COL_FILE_PATH = "File Path" # (If needed)

# Column name for data from MARCXMLProcessor
COL_TEXT_FOR_EMBEDDING = "text_for_embedding"
COL_MARC_RECORD_ID = "record_id" # (If needed)
# Add other key MARCXML output columns if they are explicitly used by name elsewhere

# Embedding Modes
EMBEDDING_MODE_LOCAL = "local"
EMBEDDING_MODE_API = "api"
EMBEDDING_MODE_BATCH = "batch"

# Prompt Styles
PROMPT_STYLE_DEFAULT = "default"
PROMPT_STYLE_REFERENCES = "references"
