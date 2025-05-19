from scripts.data_processing.email.config_loader import ConfigLoader
import os
import logging
from datetime import datetime

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = r"C:\Users\hagaybar\OneDrive - Tel-Aviv University\My Personal files\systems\AI Project\DATA\emails\logs"
    log_filename = f"{log_path}\pipeline_log_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info("📍 Pipeline logging started")

setup_logging()
test_config = ConfigLoader(config_file=r"C:\Users\hagaybar\OneDrive - Tel-Aviv University\My Personal files\systems\AI Project\config\emails\config.yaml")
data_dir = test_config.get("paths.data_dir")
email_dir =  test_config.get("paths.email_dir")
if os.path.isdir(email_dir):
    files = os.listdir(email_dir)
    logging.info(f"found Files in directory: {files}")
    #  logging.info("found Files in directory:", files)
else:
    
    logging.info(f"The path '{email_dir}' is not a valid directory.")