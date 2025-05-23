import unittest
from scripts.prompting.prompt_builder import TextFilePromptBuilder

class TestTextFilePromptBuilder(unittest.TestCase):

    def test_build_prompt_default_style(self):
        builder = TextFilePromptBuilder() # Defaults to style="default"
        query = "What is the capital of France?"
        context = "France is a country in Europe. Its capital is Paris."
        
        prompt = builder.build(query, context)
        
        self.assertIn("You are an AI assistant helping answer questions based on information from text documents.", prompt)
        self.assertIn("Use only the information provided in the context below.", prompt)
        self.assertIn("If the answer is not found at all, reply with \"I don't know.\"", prompt)
        self.assertIn("Context from text files:\n" + context, prompt)
        self.assertIn("Question: " + query, prompt)
        self.assertIn("Answer:", prompt)
        # Check that reference-specific instructions are NOT present
        self.assertNotIn("Every factual claim must be cited", prompt)

    def test_build_prompt_references_style(self):
        builder = TextFilePromptBuilder(style="references")
        query = "What are the key points?"
        context = "[1] The first key point is about apples. [2] The second concerns oranges."
        
        prompt = builder.build(query, context)
        
        self.assertIn("You are an AI assistant answering questions based on information from text documents.", prompt)
        self.assertIn("Each context chunk is labeled [1], [2], etc.", prompt)
        self.assertIn("Every factual claim must be cited** using these labels (e.g., [1], [2]).", prompt)
        self.assertIn("❗ Do not include a 'Sources' section. It will be added automatically later.", prompt)
        self.assertIn("If no answer can be derived from the context, reply: 'I don't know.'", prompt)
        self.assertIn("Context from text files:\n" + context, prompt)
        self.assertIn("Question: " + query, prompt)
        self.assertIn("Answer:", prompt)

    def test_build_dispatch_default_style_explicit(self):
        builder = TextFilePromptBuilder(style="default")
        query = "Test query"
        context = "Test context for default."
        
        # Expected prompt structure for default style
        expected_instructions = (
            "You are an AI assistant helping answer questions based on information from text documents.\n"
            "Use only the information provided in the context below. If the context contains relevant information, "
            "summarize it clearly and concisely. Even if only partial information is available, do your best to synthesize it.\n"
            "If the answer is not found at all, reply with \"I don't know.\"\n"
        )
        expected_prompt = f"""{expected_instructions}
---
Context from text files:
{context}
---
Question: {query}
Answer:"""
        
        actual_prompt = builder.build(query, context)
        self.assertEqual(actual_prompt, expected_prompt)

    def test_build_dispatch_references_style_explicit(self):
        builder = TextFilePromptBuilder(style="references")
        query = "Test query for refs"
        context = "Test context for references, like [1] this."

        # Expected prompt structure for references style
        expected_instructions = (
            "You are an AI assistant answering questions based on information from text documents.\n"
            "Each context chunk is labeled [1], [2], etc.\n\n"
            "✅ When writing your answer:\n"
            "- **Every factual claim must be cited** using these labels (e.g., [1], [2]).\n"
            "- These citations help the user verify where the information came from.\n"
            "- If you don't include any [number] citations, your answer will be considered incomplete.\n\n"
            "❗ Do not include a 'Sources' section. It will be added automatically later.\n"
            "If no answer can be derived from the context, reply: 'I don't know.'\n"
        )
        expected_prompt = f"""{expected_instructions}
---
Context from text files:
{context}
---
Question: {query}
Answer:"""
        actual_prompt = builder.build(query, context)
        self.assertEqual(actual_prompt, expected_prompt)

    def test_build_unknown_style_defaults_to_default_prompt(self):
        builder = TextFilePromptBuilder(style="some_unknown_style")
        query = "Query for unknown style"
        context = "Context for unknown style."
        
        # Expected prompt structure for default style (since unknown should default to it)
        expected_instructions = (
            "You are an AI assistant helping answer questions based on information from text documents.\n"
            "Use only the information provided in the context below. If the context contains relevant information, "
            "summarize it clearly and concisely. Even if only partial information is available, do your best to synthesize it.\n"
            "If the answer is not found at all, reply with \"I don't know.\"\n"
        )
        expected_prompt = f"""{expected_instructions}
---
Context from text files:
{context}
---
Question: {query}
Answer:"""
        
        actual_prompt = builder.build(query, context)
        self.assertEqual(actual_prompt, expected_prompt, "Unknown style should fall back to default prompt structure.")

if __name__ == '__main__':
    unittest.main()
