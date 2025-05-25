# File: scripts/ui/config_form_builder.py
import streamlit as st
from scripts.pipeline.rag_pipeline import RAGPipeline # For data type constants
from scripts.utils.constants import (
    EMBEDDING_MODE_LOCAL, EMBEDDING_MODE_API, EMBEDDING_MODE_BATCH,
    PROMPT_STYLE_DEFAULT, PROMPT_STYLE_REFERENCES
) # Assuming these are defined

class ConfigFormBuilder:
    def __init__(self, streamlit_instance, initial_config=None, mode="create"):
        self.st = streamlit_instance
        self.mode = mode # "create" or "edit"
        self.config_data = initial_config.copy() if initial_config else {}
        self.session_state_prefix = f"{self.mode}_form_" # Mode-specific prefix
        
        # Define known models for dynamic suggestions
        self.KNOWN_LOCAL_MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "other/local-model"]
        self.KNOWN_API_MODELS = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
        self.KNOWN_COMPLETION_MODELS = ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]


    def _get_key(self, key_name):
        return f"{self.session_state_prefix}{key_name}"

    def _init_session_state(self, key, default_value):
        session_key = self._get_key(key)
        if session_key not in self.st.session_state:
            self.st.session_state[session_key] = default_value

    def render_general_settings(self):
        self.st.subheader("General Settings")
        # Initialize task_name from config_data if in edit mode, else empty
        default_task_name = self.config_data.get("task_name", "")
        self._init_session_state("task_name", default_task_name)
        
        is_read_only = self.mode == "edit"
        self.st.text_input("Task Name:", key=self._get_key("task_name"), disabled=is_read_only)

    def render_embedding_settings(self):
        self.st.subheader("Embedding Settings")
        current_embedding_config = self.config_data.get("embedding", {})
        current_persisted_mode = current_embedding_config.get("mode", EMBEDDING_MODE_LOCAL)

        # Key for storing the selected embedding mode (local, api, batch) in session state
        embedding_mode_session_key = self._get_key("embedding_mode")

        # Initialize session state for "embedding_mode"
        if self.st.session_state.get(self._get_key("current_form_mode")) != self.mode or \
           embedding_mode_session_key not in self.st.session_state:
            self.st.session_state[embedding_mode_session_key] = current_persisted_mode
        self.st.session_state[self._get_key("current_form_mode")] = self.mode # Track current form mode

        embedding_mode_disabled = False
        embedding_mode_disabled_reason = None
        info_message = None
        all_modes = [EMBEDDING_MODE_LOCAL, EMBEDDING_MODE_API, EMBEDDING_MODE_BATCH]
        available_modes = all_modes # Default for "create" mode

        if self.mode == "edit":
            if current_persisted_mode == EMBEDDING_MODE_LOCAL:
                embedding_mode_disabled = True
                embedding_mode_disabled_reason = "Embedding mode is fixed to 'local' for existing tasks. To change, create a new task."
                info_message = embedding_mode_disabled_reason
                self.st.session_state[embedding_mode_session_key] = EMBEDDING_MODE_LOCAL # Ensure it's locked to local
            else: 
                available_modes = [EMBEDDING_MODE_API, EMBEDDING_MODE_BATCH]
                info_message = "Embedding mode can be switched between 'api' and 'batch'. Model name and dimension can be updated. 'local' mode is fixed for this task."
                if self.st.session_state[embedding_mode_session_key] == EMBEDDING_MODE_LOCAL:
                    self.st.session_state[embedding_mode_session_key] = current_persisted_mode
        
        selected_mode_from_session = self.st.session_state[embedding_mode_session_key]
        
        if selected_mode_from_session not in available_modes: # Final consistency check
            self.st.session_state[embedding_mode_session_key] = available_modes[0]
            selected_mode_from_session = available_modes[0]

        try:
            current_selection_index = available_modes.index(selected_mode_from_session)
        except ValueError: current_selection_index = 0

        embedding_mode_widget_key = self._get_key("embedding_mode_select")
        
        def on_embedding_mode_change():
            new_widget_mode = self.st.session_state[embedding_mode_widget_key]
            self.st.session_state[embedding_mode_session_key] = new_widget_mode
            for key_suffix in ["embedding_model_name", "embedding_model_name_select", "embedding_dim", 
                               "last_embedding_mode_for_model_reset", "last_model_for_dim_reset"]:
                self.st.session_state.pop(self._get_key(key_suffix), None)

        self.st.selectbox(
            "Embedding Mode:", available_modes, index=current_selection_index,
            key=embedding_mode_widget_key, disabled=embedding_mode_disabled,
            help=embedding_mode_disabled_reason,
            on_change=on_embedding_mode_change if not embedding_mode_disabled else None
        )
        
        effective_selected_mode = self.st.session_state[embedding_mode_session_key] # This is the true source of mode for logic below

        if info_message: self.st.info(info_message)

        model_options = self.KNOWN_LOCAL_MODELS if effective_selected_mode == EMBEDDING_MODE_LOCAL else self.KNOWN_API_MODELS
        model_name_session_key = self._get_key("embedding_model_name")
        default_model_name_from_config = current_embedding_config.get("model_name", model_options[0] if model_options else "")
        
        last_mode_for_model_reset = self.st.session_state.get(self._get_key("last_embedding_mode_for_model_reset"))

        # Initialize or reset model_name if mode changed or if it's not in current options
        if model_name_session_key not in self.st.session_state or \
           effective_selected_mode != last_mode_for_model_reset or \
           (self.st.session_state.get(model_name_session_key) not in model_options and model_options):
            
            if self.mode == "edit" and effective_selected_mode == current_persisted_mode and default_model_name_from_config in model_options:
                self.st.session_state[model_name_session_key] = default_model_name_from_config
            elif model_options: self.st.session_state[model_name_session_key] = model_options[0]
            else: self.st.session_state[model_name_session_key] = ""
        
        self.st.session_state[self._get_key("last_embedding_mode_for_model_reset")] = effective_selected_mode
        current_model_in_session = self.st.session_state.get(model_name_session_key,"") # Default to "" if key somehow missing
        
        try:
            model_idx = model_options.index(current_model_in_session) if current_model_in_session in model_options else 0
        except ValueError: model_idx = 0
        if not model_options and current_model_in_session != "": # If no options but session has a value, reset
            self.st.session_state[model_name_session_key] = ""
        
        model_name_widget_key = self._get_key("embedding_model_name_select")
        def on_model_name_change():
            self.st.session_state[model_name_session_key] = self.st.session_state[model_name_widget_key]
            self.st.session_state.pop(self._get_key("embedding_dim"), None) # Reset dim for suggestion
            self.st.session_state.pop(self._get_key("last_model_for_dim_reset"), None)

        self.st.selectbox("Model Name:", model_options, index=model_idx, key=model_name_widget_key,
                          disabled=embedding_mode_disabled, # Model name disabled if mode is fixed
                          on_change=on_model_name_change if not embedding_mode_disabled else None)

        embedding_dim_session_key = self._get_key("embedding_dim")
        embedding_dim_disabled_by_mode = (self.mode == "edit" and current_persisted_mode == EMBEDDING_MODE_LOCAL)
        
        default_dim_from_config = current_embedding_config.get("embedding_dim", 0)
        current_model_for_dim_suggestion = self.st.session_state.get(model_name_session_key, "") # Use current value
        last_model_for_dim_reset = self.st.session_state.get(self._get_key("last_model_for_dim_reset"))

        suggested_dims = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072,
                          "text-embedding-ada-002": 1536, "sentence-transformers/all-MiniLM-L6-v2": 384}
        
        # Initialize or update dimension
        if embedding_dim_session_key not in self.st.session_state or \
           current_model_for_dim_suggestion != last_model_for_dim_reset: # This condition means model changed
            suggested_dim = suggested_dims.get(current_model_for_dim_suggestion)
            if not embedding_dim_disabled_by_mode and suggested_dim is not None:
                self.st.session_state[embedding_dim_session_key] = suggested_dim
            # If editing an existing config, prefer its dimension if not suggesting a new one
            elif self.mode == "edit" and default_dim_from_config > 0 and not (not embedding_dim_disabled_by_mode and suggested_dim is not None) :
                 self.st.session_state[embedding_dim_session_key] = default_dim_from_config
            elif embedding_dim_disabled_by_mode : # Fixed to local mode during edit
                 self.st.session_state[embedding_dim_session_key] = suggested_dims.get("sentence-transformers/all-MiniLM-L6-v2", 384)
            else: # Create mode, no specific suggestion based on model, or no pre-existing valid dim
                 self.st.session_state[embedding_dim_session_key] = 0 # Let validation catch it if it's required, or set a generic default like 1536 for API

        self.st.session_state[self._get_key("last_model_for_dim_reset")] = current_model_for_dim_suggestion
        
        self.st.number_input("Embedding Dimension:", min_value=1, step=1, key=embedding_dim_session_key, 
                             disabled=embedding_dim_disabled_by_mode)
        if embedding_dim_disabled_by_mode:
            self.st.caption("Embedding dimension is fixed for 'local' mode in existing tasks.")


    def render_chunking_settings(self):
        self.st.subheader("Chunking Settings")
        cfg = self.config_data.get("chunking", {})
        self._init_session_state("max_chunk_size", cfg.get("max_chunk_size", 500))
        self._init_session_state("overlap", cfg.get("overlap", 50))
        self._init_session_state("min_chunk_size", cfg.get("min_chunk_size", 150))
        self._init_session_state("similarity_threshold", cfg.get("similarity_threshold", 0.8))
        self._init_session_state("language_model", cfg.get("language_model", "en_core_web_sm"))
        self._init_session_state("chunking_embedding_model", cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))

        self.st.number_input("Max Chunk Size (tokens):", min_value=1, step=10, key=self._get_key("max_chunk_size"))
        self.st.number_input("Overlap (tokens):", min_value=0, step=5, key=self._get_key("overlap"))
        self.st.number_input("Min Chunk Size (chars):", min_value=1, step=10, key=self._get_key("min_chunk_size"))
        self.st.slider("Similarity Threshold:", 0.0, 1.0, step=0.05, key=self._get_key("similarity_threshold"))
        self.st.text_input("SpaCy Language Model:", key=self._get_key("language_model"))
        self.st.text_input("Chunking Embedding Model (for similarity):", key=self._get_key("chunking_embedding_model"))

    def render_data_sources_settings(self):
        self.st.subheader("Data Source Settings")
        # In a real scenario, this would be more dynamic, allowing adding/removing sources.
        # For now, assume we configure one type or show all.
        
        # Using RAGPipeline constants for data types
        supported_types = RAGPipeline.SUPPORTED_DATA_TYPES
        
        # Allow selecting which data sources to configure
        # For "create" mode, default to empty list. For "edit", pre-select based on current_config.
        default_selected_sources = [
            dt for dt in supported_types 
            if RAGPipeline.get_config_section_for_data_type(dt) in self.config_data
        ] if self.mode == "edit" else []
        
        self._init_session_state("selected_data_sources", default_selected_sources)

        selected_sources = self.st.multiselect(
            "Select data types to configure:",
            options=supported_types,
            default=self.st.session_state[self._get_key("selected_data_sources")],
            key=self._get_key("selected_data_sources_multiselect")
        )
        self.st.session_state[self._get_key("selected_data_sources")] = selected_sources


        ds_config = self.config_data # Main config for data sources
        
        if RAGPipeline.DATA_TYPE_EMAIL in selected_sources:
            self.st.markdown("--- \n**Email Source (Outlook):**")
            email_cfg = ds_config.get("outlook", {})
            self._init_session_state("outlook_account_name", email_cfg.get("account_name", ""))
            self._init_session_state("outlook_folder_path", email_cfg.get("folder_path", "Inbox"))
            self._init_session_state("outlook_days_to_fetch", email_cfg.get("days_to_fetch", 7))
            self.st.text_input("Outlook Account Name:", key=self._get_key("outlook_account_name"))
            self.st.text_input("Folder Path (e.g., Inbox > Subfolder):", key=self._get_key("outlook_folder_path"))
            self.st.number_input("Days to Fetch:", min_value=1, step=1, key=self._get_key("outlook_days_to_fetch"))

        if RAGPipeline.DATA_TYPE_TEXT in selected_sources:
            self.st.markdown("--- \n**Text Files Source:**")
            text_cfg = ds_config.get("text_files", {})
            self._init_session_state("text_files_input_dir", text_cfg.get("input_dir", "data/text_files/[task_name]"))
            self.st.text_input("Input Directory (Text Files):", key=self._get_key("text_files_input_dir"))

        if RAGPipeline.DATA_TYPE_XML in selected_sources:
            self.st.markdown("--- \n**XML Files Source:**")
            xml_cfg = ds_config.get("xml_files", {})
            self._init_session_state("xml_files_input_dir", xml_cfg.get("input_dir", "data/xml_files/[task_name]"))
            self.st.text_input("Input Directory (XML Files):", key=self._get_key("xml_files_input_dir"))
            
        if RAGPipeline.DATA_TYPE_MARCXML in selected_sources:
            self.st.markdown("--- \n**MARCXML Files Source:**")
            marcxml_cfg = ds_config.get("marcxml_files", {})
            self._init_session_state("marcxml_files_input_dir", marcxml_cfg.get("input_dir", "data/marcxml_files/[task_name]"))
            self.st.text_input("Input Directory (MARCXML Files):", key=self._get_key("marcxml_files_input_dir"))


    def render_retrieval_settings(self):
        self.st.subheader("Retrieval Settings")
        cfg = self.config_data.get("retrieval", {})
        self._init_session_state("retrieval_top_k", cfg.get("top_k", 5))
        self.st.number_input("Top-K Chunks to Retrieve:", min_value=1, step=1, key=self._get_key("retrieval_top_k"))

    def render_generation_settings(self):
        self.st.subheader("Answer Generation Settings")
        cfg = self.config_data.get("generation", {})
        prompt_cfg = self.config_data.get("prompting", {}) # Prompt style is separate

        self._init_session_state("generation_model", cfg.get("model", "gpt-4o-mini"))
        self._init_session_state("prompt_style", prompt_cfg.get("style", PROMPT_STYLE_DEFAULT))
        
        self.st.selectbox("OpenAI Model for Generation:", self.KNOWN_COMPLETION_MODELS, 
                          index=self.KNOWN_COMPLETION_MODELS.index(self.st.session_state[self._get_key("generation_model")]) if self.st.session_state[self._get_key("generation_model")] in self.KNOWN_COMPLETION_MODELS else 0,
                          key=self._get_key("generation_model"))
        
        prompt_styles = [PROMPT_STYLE_DEFAULT, PROMPT_STYLE_REFERENCES]
        self.st.selectbox("Prompt Style:", prompt_styles, 
                          index=prompt_styles.index(self.st.session_state[self._get_key("prompt_style")]) if self.st.session_state[self._get_key("prompt_style")] in prompt_styles else 0,
                          key=self._get_key("prompt_style"))

    def get_config_data(self):
        # Collects data from st.session_state and structures it into a config dictionary
        # This needs to carefully mirror the expected YAML structure.
        
        # Helper to get value from session_state, or None if not found
        def _get_ss_val(key_suffix):
            return self.st.session_state.get(self._get_key(key_suffix))

        data = {}
        task_name_val = _get_ss_val("task_name")
        if not task_name_val and self.mode == "create": # Only enforce for create mode here
            raise ValueError("Task Name cannot be empty.")
        data["task_name"] = task_name_val
        
        embedding_model_name_val = _get_ss_val("embedding_model_name")
        if not embedding_model_name_val:
            raise ValueError("Embedding Model Name cannot be empty.")
        embedding_dim_val = _get_ss_val("embedding_dim")
        if not embedding_dim_val or embedding_dim_val <=0:
            raise ValueError("Embedding Dimension must be greater than 0.")

        data["embedding"] = {
            "mode": _get_ss_val("embedding_mode"),
            "model_name": embedding_model_name_val,
            "embedding_dim": embedding_dim_val,
            # Default output_dir, index_filename, metadata_filename will be set by RAGPipeline.configure_task
        }
        data["chunking"] = {
            "max_chunk_size": _get_ss_val("max_chunk_size"),
            "overlap": _get_ss_val("overlap"),
            "min_chunk_size": _get_ss_val("min_chunk_size"),
            "similarity_threshold": _get_ss_val("similarity_threshold"),
            "language_model": _get_ss_val("language_model"),
            "embedding_model": _get_ss_val("chunking_embedding_model"),
        }

        selected_sources = _get_ss_val("selected_data_sources") or []
        if not selected_sources and self.mode=="create": # Require at least one source for new tasks
             raise ValueError("At least one data source must be selected and configured.")

        if RAGPipeline.DATA_TYPE_EMAIL in selected_sources:
            outlook_account_name = _get_ss_val("outlook_account_name")
            outlook_folder_path = _get_ss_val("outlook_folder_path")
            if not outlook_account_name: raise ValueError("Outlook Account Name cannot be empty if Email source is selected.")
            if not outlook_folder_path: raise ValueError("Outlook Folder Path cannot be empty if Email source is selected.")
            data["outlook"] = {
                "account_name": outlook_account_name,
                "folder_path": outlook_folder_path,
                "days_to_fetch": _get_ss_val("outlook_days_to_fetch"),
            }
        if RAGPipeline.DATA_TYPE_TEXT in selected_sources:
            text_input_dir = _get_ss_val("text_files_input_dir")
            if not text_input_dir: raise ValueError("Input Directory (Text Files) cannot be empty if Text Files source is selected.")
            data["text_files"] = {"input_dir": text_input_dir}
        if RAGPipeline.DATA_TYPE_XML in selected_sources:
            xml_input_dir = _get_ss_val("xml_files_input_dir")
            if not xml_input_dir: raise ValueError("Input Directory (XML Files) cannot be empty if XML Files source is selected.")
            data["xml_files"] = {"input_dir": xml_input_dir}
        if RAGPipeline.DATA_TYPE_MARCXML in selected_sources:
            marcxml_input_dir = _get_ss_val("marcxml_files_input_dir")
            if not marcxml_input_dir: raise ValueError("Input Directory (MARCXML Files) cannot be empty if MARCXML Files source is selected.")
            data["marcxml_files"] = {"input_dir": marcxml_input_dir}

        data["retrieval"] = {"top_k": _get_ss_val("retrieval_top_k")}
        data["generation"] = {"model": _get_ss_val("generation_model")}
        data["prompting"] = {"style": _get_ss_val("prompt_style")}
        
        # Paths are largely handled by RAGPipeline.configure_task
        # but we can provide placeholders or let configure_task fill them.
        data["paths"] = {} # Let configure_task handle path injection

        return data

    def display_form(self):
        self.render_general_settings()
        self.render_embedding_settings()
        self.render_chunking_settings()
        self.render_data_sources_settings()
        self.render_retrieval_settings()
        self.render_generation_settings()
