import os
import time
import json
from typing import List, Dict, Optional
from openai import OpenAI


class BatchEmbedder:
    """
    Handles OpenAI batch embedding workflow:
    - Writes input chunks to JSONL
    - Submits batch job
    - Polls for status
    - Downloads and parses result
    - Returns {custom_id: embedding} dict
    """

    def __init__(self, model: str = "text-embedding-3-small", output_dir: str = "batch_outputs", api_key: Optional[str] = None):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.client = OpenAI(api_key=api_key or os.getenv("OPEN_AI"))

    def prepare_jsonl_file(self, texts: List[str]) -> str:
        """Write batch embedding requests to a JSONL file."""
        jsonl_path = os.path.join(self.output_dir, "batch_input.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(texts):
                request = {
                    "custom_id": f"chunk-{i}",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": self.model,
                        "input": text
                    }
                }
                f.write(json.dumps(request) + "\n")
        return jsonl_path

    def submit_batch(self, jsonl_path: str) -> str:
        """Uploads the JSONL file and submits a batch embedding job."""
        uploaded = self.client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
        batch = self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={"description": "Batch embedding job"}
        )
        return batch.id

    def wait_for_completion(self, batch_id: str, poll_interval: int = 10) -> Dict:
        """Polls batch status until it's completed or failed."""
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            if status in ["completed", "failed", "cancelled", "expired"]:
                return batch
            time.sleep(poll_interval)

    def download_output(self, output_file_id: str) -> List[Dict]:
        """Downloads and parses the output JSONL file."""
        response = self.client.files.content(output_file_id)
        lines = response.text.strip().splitlines()
        return [json.loads(line) for line in lines]

    def extract_embeddings(self, results: List[Dict]) -> Dict[str, List[float]]:
        """Returns a mapping of custom_id → embedding vector."""
        output = {}
        for item in results:
            custom_id = item["custom_id"]
            embedding = item["response"]["body"]["data"][0]["embedding"]
            output[custom_id] = embedding
        return output

    def run(self, texts: List[str]) -> Dict[str, List[float]]:
        """Full batch embedding flow."""
        jsonl_path = self.prepare_jsonl_file(texts)
        batch_id = self.submit_batch(jsonl_path)
        print(f"🚀 Batch submitted. ID: {batch_id}")
        batch = self.wait_for_completion(batch_id)

        if batch.status != "completed":
            raise RuntimeError(f"Batch failed with status: {batch.status}")

        output_file_id = batch.output_file_id
        results = self.download_output(output_file_id)
        return self.extract_embeddings(results)
