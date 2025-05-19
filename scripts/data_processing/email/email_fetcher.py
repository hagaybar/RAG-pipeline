import os
import win32com.client as win32
import pandas as pd
from datetime import datetime, timedelta
from scripts.utils.logger import LoggerManager


class EmailFetcher:
    def __init__(self, config: dict):
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
        try:
            outlook = win32.Dispatch("Outlook.Application").GetNamespace("MAPI")
            self.logger.info("✅ Connected to Outlook.")
            return outlook
        except Exception as e:
            self.logger.info(f"❌ Outlook connection failed: {e}")
            raise

    def fetch_emails_from_folder(self, return_dataframe=False, save=True):
        outlook = self.connect_to_outlook()
        account_folder = self._get_account_folder(outlook)
        target_folder = self._get_target_folder(account_folder)

        cutoff = datetime.now() - timedelta(days=self.days)
        filtered_items = target_folder.Items.Restrict(
            f"[ReceivedTime] >= '{cutoff.strftime('%m/%d/%Y %H:%M %p')}'"
        )

        email_data = []
        for item in filtered_items:
            if hasattr(item, "Class") and item.Class == 43:
                email_data.append({
                    "Subject": item.Subject,
                    "Sender": item.SenderName,
                    "Received": item.ReceivedTime.strftime("%Y-%m-%d %H:%M:%S"),
                    "Raw Body": item.Body if hasattr(item, "Body") else "No Body",
                    "Cleaned Body": self.clean_email_body(item.Body)
                })

        df = pd.DataFrame(email_data)
        self.logger.info(f"Fetched {len(df)} emails from {self.folder_path}")

        if save and not df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(self.output_dir, f"{timestamp}_{self.output_file}")
            df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
            self.logger.info(f"Saved to: {out_path}")
            if not return_dataframe:
                return out_path

        return df if return_dataframe else None

    def _get_account_folder(self, outlook):
        for i in range(outlook.Folders.Count):
            folder = outlook.Folders.Item(i + 1)
            if folder.Name == self.account_name:
                return folder
        raise ValueError(f"Account '{self.account_name}' not found")

    def _get_target_folder(self, account_folder):
        folder = account_folder.Folders["Inbox"]
        for name in self.folder_path.split(">"):
            folder = folder.Folders[name]
        return folder

    def clean_email_body(self, body):
        return body  # placeholder for actual cleaning logic
