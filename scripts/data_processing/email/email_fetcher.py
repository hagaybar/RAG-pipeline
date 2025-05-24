"""
This module defines the `EmailFetcher` class, used for retrieving emails
from Microsoft Outlook. It connects to a specified account, navigates to a
target folder, fetches emails within a defined number of past days,
and can save them as a TSV or return a DataFrame. Includes basic email body
cleaning functionality.
"""
import os
import win32com.client as win32
import pandas as pd
from datetime import datetime, timedelta
import pythoncom # Added for COM initialization
import threading # Added for detailed logging
from scripts.utils.logger import LoggerManager


class EmailFetcher:
    """
    Connects to Microsoft Outlook to fetch emails from a specified account and folder.

    This class uses the `win32com.client` library to interact with the local
    Outlook application. It is initialized with a configuration dictionary
    (`config`) which supplies necessary parameters such as:
    - `outlook.account_name`: The Outlook account to use.
    - `outlook.folder_path`: The specific folder path within the account
      (e.g., "Inbox" or "Inbox > MySubfolder").
    - `outlook.days_to_fetch`: The number of past days from which to retrieve emails.
    - `paths.email_dir`: Directory to save fetched email data.
    - `paths.output_file`: Filename for the saved TSV.

    The core functionality is in `fetch_emails_from_folder`, which handles
    connecting to Outlook, navigating to the target folder, filtering emails
    based on the `days_to_fetch` criterion, and extracting relevant data
    (Subject, Sender, Received Time, Body) into a Pandas DataFrame.
    A basic `clean_email_body` method is included as a placeholder for
    text preprocessing. The resulting DataFrame can be saved as a TSV file or
    returned by the method. Logging is performed using a `LoggerManager` instance.
    """
    def __init__(self, config: dict):
        """
        Initializes the EmailFetcher instance.

        Args:
            config (dict): A configuration dictionary containing settings for
                           the Outlook account (config["outlook"]), paths for
                           saving data (config.get("paths", {})), and logging.
                           Extracts account_name, folder_path, days_to_fetch,
                           output_dir, output_file, and log_dir. Initializes a
                           logger and creates output directories if they don't exist.
        """
        self.config = config
        self.logger = LoggerManager.get_logger("EmailFetcher")
        self.account_name = config["outlook"]["account_name"]
        self.folder_path = config["outlook"]["folder_path"]
        self.days = config["outlook"].get("days_to_fetch", 1)

        self.output_dir = config.get("paths", {}).get("email_dir", "data/cleaned")
        self.output_file = config.get("paths", {}).get("output_file", "emails.tsv")
        self.log_dir = config.get("paths", {}).get("log_dir", "logs/email_fetching")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.output_full_path = os.path.join(self.output_dir, self.output_file)

    # def log_action(self, message, logfile_name="email_processing.log"):
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     logfile_path = os.path.join(self.log_dir, f"{timestamp}_{logfile_name}")
    #     with open(logfile_path, "a", encoding="utf-8") as log:
    #         log.write(f"{datetime.now()} - {message}\n")

    def connect_to_outlook(self):
        """
        Establishes a connection to the Microsoft Outlook application.

        Uses `win32com.client` to dispatch an Outlook Application COM object
        and get the MAPI namespace.

        Args:
            None.

        Returns:
            win32com.client.Dispatch: An Outlook MAPI namespace object if successful.

        Raises:
            Exception: If the connection to Outlook fails for any reason.
        """
        try:
            outlook = win32.Dispatch("Outlook.Application").GetNamespace("MAPI")
            self.logger.info("✅ Connected to Outlook.")
            return outlook
        except Exception as e:
            self.logger.info(f"❌ Outlook connection failed: {e}")
            raise

    def fetch_emails_from_folder(self, return_dataframe: bool = False, save: bool = True):
        """
        Fetches emails from the configured Outlook folder, filters by date,
        extracts data, and optionally saves to TSV or returns a DataFrame.

        The process involves:
        1. Connecting to Outlook.
        2. Navigating to the specified account and target folder.
        3. Filtering emails based on `self.days` (number of past days).
        4. Extracting Subject, Sender, Received Time, Raw Body, and Cleaned Body.
        5. Optionally saving the data to a timestamped TSV file.
        6. Optionally returning the data as a Pandas DataFrame.

        Args:
            return_dataframe (bool, optional): If True, returns the fetched emails
                as a Pandas DataFrame. Defaults to False.
            save (bool, optional): If True and emails are fetched, saves the emails
                to a timestamped TSV file in the configured output directory.
                Defaults to True.

        Returns:
            pd.DataFrame | str | None:
                - If `return_dataframe` is True, returns a Pandas DataFrame of the email data.
                - If `save` is True and `return_dataframe` is False, returns the
                  path (str) to the saved TSV file.
                - Otherwise, returns `None`.
                - Returns `None` or an empty DataFrame if no emails are fetched.
        """
        thread_id = threading.get_ident()
        self.logger.info(f"[EmailFetcher][Thread: {thread_id}] Entering fetch_emails_from_folder.")
        
        com_initialized_in_this_call = False
        hr_init_result = None

        try:
            self.logger.info(f"[EmailFetcher][Thread: {thread_id}] Attempting pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED).")
            hr_init_result = pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
            # S_OK (0) means initialized successfully by this call.
            # S_FALSE (1) means already initialized, possibly by an outer call or earlier on this thread.
            # RPC_E_CHANGED_MODE means thread already init'd with different model.
            self.logger.info(f"[EmailFetcher][Thread: {thread_id}] CoInitializeEx call completed. HRESULT: {hr_init_result}. Interpretation: 0=OK, 1=S_FALSE (already init'd), other=Error.")
            if hr_init_result == 0 or hr_init_result == 1: # S_OK or S_FALSE
                 com_initialized_in_this_call = True # Consider S_FALSE as "safe to proceed and we should uninit"
            # If CoInitializeEx raises an error (e.g. RPC_E_CHANGED_MODE), it will be caught by the outer except block.

            self.logger.info(f"[EmailFetcher][Thread: {thread_id}] Proceeding with Outlook COM operations.")
            outlook = self.connect_to_outlook()
            account_folder = self._get_account_folder(outlook)
            target_folder = self._get_target_folder(account_folder)

            cutoff = datetime.now() - timedelta(days=self.days)
            self.logger.debug(f"[EmailFetcher][Thread: {thread_id}] Filtering items received after: {cutoff.strftime('%m/%d/%Y %H:%M %p')}")
            
            restricted_items_call_successful = False
            try:
                filtered_items = target_folder.Items.Restrict(f"[ReceivedTime] >= '{cutoff.strftime('%m/%d/%Y %H:%M %p')}'")
                restricted_items_call_successful = True
            except pythoncom.com_error as e_restrict: # Catch specific COM errors
                 self.logger.error(f"[EmailFetcher][Thread: {thread_id}] pythoncom.com_error during Items.Restrict: HRESULT={e_restrict.hresult}, Message={e_restrict.strerror}")
                 raise # Re-raise to be caught by the outer try-except for general error message
            except Exception as e_restrict_generic:
                 self.logger.error(f"[EmailFetcher][Thread: {thread_id}] Generic error during Items.Restrict: {str(e_restrict_generic)}")
                 raise

            if not restricted_items_call_successful: # Should not be reached if error is raised
                self.logger.warning(f"[EmailFetcher][Thread: {thread_id}] Items.Restrict call did not succeed (though no exception was caught - unusual).")
                # Handle empty or problematic filtered_items if necessary
                filtered_items = []


            email_data = []
            self.logger.info(f"[EmailFetcher][Thread: {thread_id}] Iterating through {len(filtered_items) if restricted_items_call_successful else 'UNKNOWN number of'} items...")
            for item_count, item in enumerate(filtered_items):
                if hasattr(item, "Class") and item.Class == 43: # olMail
                    if item_count < 3: # Log details for first 3 items
                        self.logger.debug(f"[EmailFetcher][Thread: {thread_id}] Processing item {item_count + 1} - EntryID: {getattr(item, 'EntryID', 'N/A')} Subject: {getattr(item, 'Subject', 'N/A')}")
                    email_data.append({
                        "EntryID": item.EntryID,
                        "Subject": item.Subject,
                        "Sender": item.SenderName,
                        "Received": item.ReceivedTime.strftime("%Y-%m-%d %H:%M:%S"),
                        "Raw Body": item.Body if hasattr(item, "Body") else "No Body",
                        "Cleaned Body": self.clean_email_body(item.Body)
                    })
            
            df = pd.DataFrame(email_data)
            self.logger.info(f"[EmailFetcher][Thread: {thread_id}] Fetched {len(df)} emails from {self.folder_path}.")

            if save and not df.empty:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(self.output_dir, f"{timestamp}_{self.output_file}")
                df.to_csv(out_path, sep="	", index=False, encoding="utf-8") # Note: Using literal tab for sep
                self.logger.info(f"[EmailFetcher][Thread: {thread_id}] Saved to: {out_path}")
                if not return_dataframe:
                    return out_path
            
            return df if return_dataframe else None

        except pythoncom.com_error as e_com:
            # This will catch COM errors from CoInitializeEx itself or any subsequent COM operation
            self.logger.error(f"[EmailFetcher][Thread: {thread_id}] pythoncom.com_error occurred: HRESULT={e_com.hresult}, Message={e_com.strerror}, FullError={e_com}. HRESULT from CoInitializeEx was: {hr_init_result}")
            # The original error was (-2147221008, 'CoInitialize has not been called.')
            # This HRESULT -2147221008 (0x800401F0) is CO_E_NOTINITIALIZED.
            raise  # Re-raise the com_error to be potentially caught by RAGPipeline and displayed in UI
        except Exception as e_generic:
            self.logger.error(f"[EmailFetcher][Thread: {thread_id}] An unexpected error occurred in fetch_emails_from_folder: {str(e_generic)}. HRESULT from CoInitializeEx was: {hr_init_result}")
            raise # Re-raise generic error
        finally:
            final_thread_id = threading.get_ident()
            self.logger.info(f"[EmailFetcher][Thread: {final_thread_id}] In finally block of fetch_emails_from_folder.")
            if com_initialized_in_this_call: # Only uninitialize if CoInitializeEx seemed to succeed (returned 0 or 1)
                try:
                    pythoncom.CoUninitialize()
                    self.logger.info(f"[EmailFetcher][Thread: {final_thread_id}] CoUninitialize called.")
                except pythoncom.com_error as e_uninit_com: # Catch specific error during uninitialize
                    self.logger.error(f"[EmailFetcher][Thread: {final_thread_id}] pythoncom.com_error during CoUninitialize: HRESULT={e_uninit_com.hresult}, Message={e_uninit_com.strerror}")
                except Exception as e_uninit_generic:
                    self.logger.error(f"[EmailFetcher][Thread: {final_thread_id}] Generic exception during CoUninitialize: {str(e_uninit_generic)}")
            else:
                self.logger.info(f"[EmailFetcher][Thread: {final_thread_id}] CoUninitialize skipped because CoInitializeEx did not complete successfully or indicated it was not needed for this call's scope (HRESULT: {hr_init_result}).")

    def _get_account_folder(self, outlook):
        """
        Retrieves the specified Outlook account folder object.

        Iterates through the top-level folders in the Outlook namespace to find
        the one matching `self.account_name`.

        Args:
            outlook (win32com.client.Dispatch): The Outlook MAPI namespace object.

        Returns:
            win32com.client.Dispatch: The account folder object.

        Raises:
            ValueError: If the account specified in `self.account_name` is not found.
        """
        for i in range(outlook.Folders.Count):
            folder = outlook.Folders.Item(i + 1)
            if folder.Name == self.account_name:
                return folder
        raise ValueError(f"Account '{self.account_name}' not found")

    def _get_target_folder(self, account_folder):
        """
        Navigates to and returns the target email folder object within the
        given account folder, based on `self.folder_path`.

        The `self.folder_path` is expected to be a string like "Inbox" or
        "Inbox > Subfolder > AnotherSubfolder".

        Args:
            account_folder (win32com.client.Dispatch): The Outlook account folder object.

        Returns:
            win32com.client.Dispatch: The target email folder object.

        Raises:
            Exception: If any part of the `self.folder_path` is not found (propagated
                       from accessing `folder.Folders[name]`).
        """
        folder = account_folder.Folders["Inbox"]
        for name in self.folder_path.split(">"):
            folder = folder.Folders[name]
        return folder

    def clean_email_body(self, body: str) -> str:
        """
        Placeholder method for cleaning the text content of an email body.

        Currently, this method returns the email body as is. It is intended
        to be a point for future enhancements where specific cleaning logic
        (e.g., removing signatures, reply chains, HTML tags if not plain text)
        can be implemented.

        Args:
            body (str): The raw text content of the email body.

        Returns:
            str: The cleaned email body text (currently, the original body).
        """
        return body  # placeholder for actual cleaning logic
