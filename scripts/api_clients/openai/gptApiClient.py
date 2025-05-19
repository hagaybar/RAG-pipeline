import os
import logging
import tiktoken
from typing import List, Optional
from openai import OpenAI

class APIClient:
    """Handles API requests, cost estimation, and budget tracking."""

    MODEL_COSTS = {
            # all price values USD per 1M tokens
            # all price values are cut to half for batched requests
            # For embedding models where input/output is the same: 
            "text-embedding-3-small" : {"price": 0.02},
            "text-embedding-3-large" : {"price": 0.13},
            "text-embedding-ada-002" : {"price": 0.10},
            # For models with separate input and output pricing:
            "gpt-4o-mini": {"input": 0.15, "output": 0.6, "cached_input": 0.075},
            
    }


    def __init__(self, 
                 api_key: Optional[str] = None, 
                 budget_limit: float = 1.0,
                 config: Optional[dict] = None,
                 log_file: str = "api_usage.log"):
            """
            Initialize the API client with budget control and logging.

            Args:
                api_key (str, optional): OpenAI API key (defaults to environment variable OPEN_AI).
                budget_limit (float): Maximum allowed spending (default is $1.00).
                log_file (str): File path for logging API usage.
            """
            self.api_key = api_key or os.getenv("OPEN_AI")
            if not self.api_key:
                raise ValueError("No API key provided and 'OPEN_AI' environment variable is not set.")
            
            # Read from config or fall back to default
            self.embedding_cfg = config.get("embedding", {}) if config else {}
            self.embedding_dim = self.embedding_cfg.get("embedding_dim", 1536)  # Default to 1536 if not specified
            self.budget_limit = budget_limit
            self.spent = 0.0

            # Set up logging
            self.logger = logging.getLogger("APIClient")
            self.logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter("%(asctime)s - %(message)s")
            file_handler.setFormatter(formatter)
            if not self.logger.handlers:
                self.logger.addHandler(file_handler)
            
            self.logger.info(f"APIClient initialized with budget limit: ${self.budget_limit:.2f}")
 
    def count_tokens(self, text: str, model: str) -> int:
        """Counts tokens in text using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(model)
            self.logger.debug(f"for model {model}: encoding = {encoding} ; length = {len(encoding.encode(text))}")
            return len(encoding.encode(text))
        except Exception:
            return len(text.split())  # Approximate fallback for unknown models.

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int, batch: bool = False) -> float:
        """Calculates API request cost based on token usage and model pricing."""
        if model not in self.MODEL_COSTS:
            raise ValueError(f"Unknown model: {model}")

        model_info = self.MODEL_COSTS[model]
        if "price" in model_info:
            cost_per_token = model_info["price"] / 1_000_000
            total_cost = (input_tokens + output_tokens) * cost_per_token
        else:
            input_cost_per_token = model_info["input"] / 1_000_000
            output_cost_per_token = model_info["output"] / 1_000_000
            total_cost = (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)

        if batch:
            total_cost *= 0.5  # Apply batch discount

        
        self.logger.debug(f"Estimated cost for model {model}: ${total_cost:.6f} "
                  f"(Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Batch: {batch})")
        return total_cost

    def get_available_budget(self) -> float:
        """Returns the remaining available budget."""
        return self.budget_limit - self.spent

    def update_budget(self, spent: float) -> None:
        """Updates the budget after an API request."""
        self.spent += spent
        self.logger.info(f"Updated budget: ${self.get_available_budget():.2f}")

    def should_proceed(self, estimated_cost: float) -> bool:
        """Checks if a request should proceed based on the available budget."""
        available_budget = self.get_available_budget()
        cost_percent = (estimated_cost / available_budget) * 100

        if cost_percent < 20:
            return True  # Auto-execute
        elif cost_percent < 90:
            response = input(f"Request cost is ${estimated_cost:.2f} ({cost_percent:.1f}% of budget). Proceed? (y/n): ")
            return response.lower() == 'y'
        else:
            print(f"Request blocked. Estimated cost ${estimated_cost:.2f} exceeds 90% of budget.")
            return False

    def send_completion_request(self, prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 250) -> Optional[str]:
        """
        Executes an OpenAI completion request after budget validation.

        Args:
            prompt (str): The user input prompt.
            model (str): The model to use (default: gpt-4o-mini).
            max_tokens (int): Maximum output tokens.

        Returns:
            str: The API response text (if successful), or None if blocked.
        """
        input_tokens = self.count_tokens(prompt, model)
        estimated_cost = self.calculate_cost(model, input_tokens, max_tokens)

        if not self.should_proceed(estimated_cost):
            return None  # Request blocked due to budget

        client = OpenAI(api_key=self.api_key,)
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=max_tokens
            )

            # Extract actual usage details
            usage = response.usage
            actual_input_tokens = usage.prompt_tokens
            actual_output_tokens = usage.completion_tokens
            actual_cost = self.calculate_cost(model, actual_input_tokens, actual_output_tokens)

            # Update budget
            self.update_budget(actual_cost)

            # Log usage
            self.log_usage(model, actual_input_tokens, actual_output_tokens, actual_cost)

            return response.choices[0].message.content  # Return the model's response
        except Exception as e:
            self.logger.error(f"Error during API request: {e}")
            return None


    def get_embedding(self, text: str, model: str = "text-embedding-ada-002") -> Optional[List[float]]:
        """
        Retrieve the embedding for a given text using the specified model.

        Args:
            text (str): The input text to embed.
            model (str): The embedding model to use (default: text-embedding-ada-002).

        Returns:
            Optional[List[float]]: The embedding vector if successful, or None if the request is blocked or fails.
        """
        input_tokens = self.count_tokens(text, model)
        estimated_cost = self.calculate_cost(model, input_tokens, 0)
        if not self.should_proceed(estimated_cost):
            return None

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.embeddings.create(
                input=text,
                model=model
            )
            # 🔹 Fix: Properly extract embeddings
            embedding = response.data[0].embedding  
            # Since embeddings typically cost based on input tokens only:
            actual_cost = self.calculate_cost(model, input_tokens, 0)
            self.update_budget(actual_cost)
            self.logger.info(f"Retrieved embedding for text with {input_tokens} tokens at cost ${actual_cost:.6f}.")
            return embedding
        except Exception as e:
            self.logger.error(f"Error retrieving embedding: {e}")
            return None



    def embed(self, texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
        """
        Embed a list of texts using get_embedding(). Compatible with GeneralPurposeEmbedder.

        Args:
            texts (List[str]): A list of strings to embed.
            model (str): Embedding model to use.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text, model=model)
            if embedding:
                embeddings.append(embedding)
            else:
                embeddings.append([0.0] * self.embedding_dim)   # fallback
        return embeddings





    def log_usage(self, model: str, input_tokens: int, output_tokens: int, cost: float) -> None:
        """Logs API usage details."""
        log_message = f"Model: {model}, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Cost: ${cost:.6f}"
        self.logger.info(log_message)



# Run the function
if __name__ == "__main__":
    # Replace with your actual API key and budget. 
    BUDGET_LIMIT = 0.01 # e.g., $1.00 total allowed spending. 
    client = APIClient()
    answer = client.send_completion_request("Please list the authors of the first encyclopedia.", model="gpt-4o-mini", max_tokens=125)
    print(f"API Response: {answer}")
    print(f"Remaining Budget: ${client.get_available_budget():.2f}")
