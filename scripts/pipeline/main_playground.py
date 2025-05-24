import argparse
import sys
import os
from scripts.pipeline.rag_pipeline import RAGPipeline # Assuming execution from project root

def main():
    parser = argparse.ArgumentParser(description="Run the RAG Email Pipeline with a specific configuration and query.")
    parser.add_argument("--config", required=True, help="Path to the task configuration YAML file")
    parser.add_argument("--query", required=True, help="The query string for the RAG pipeline")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Attempting to load pipeline with configuration: {args.config}")
        pipeline = RAGPipeline(config_path=args.config)
        
        print(f"Pipeline loaded. Attempting to run with query: '{args.query}'")
        result = pipeline.run_full_pipeline(query=args.query)
        
        print("\n=== Pipeline Result ===")
        if result is not None:
            print(result)
        else:
            print("The pipeline did not return a result. Please check logs for more details.")
            
    except FileNotFoundError as fnf_error:
        print(f"Error: A required file was not found. Details: {fnf_error}", file=sys.stderr)
        sys.exit(1)
    except ValueError as val_error:
        print(f"Error: A value error occurred, often due to incorrect configuration. Details: {val_error}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Generic error catcher for other unexpected issues
        print(f"An unexpected error occurred during pipeline execution: {type(e).__name__} - {e}", file=sys.stderr)
        # For debugging, you might want to print the full traceback
        # import traceback
        # traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # This ensures that the main function is called only when the script is executed directly
    # and not when imported as a module.
    main()