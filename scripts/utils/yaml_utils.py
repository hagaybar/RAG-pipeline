"""
This module offers utilities for fine-tuning YAML serialization.
It includes `SmartQuotedStringDumper` for context-aware string quoting (quoting
string values but not keys or numbers), and `enforce_scalar_types` to convert
numeric-looking strings to actual numbers, enhancing YAML readability and type
fidelity.
"""
# scripts/utils/yaml_utils.py
import yaml

class SmartQuotedStringDumper(yaml.SafeDumper):
    """
    Dumper that quotes string values but leaves keys and numbers unquoted.
    """

    def represent_str(self, data: str):
        """
        Overrides default string representation to selectively quote strings.

        This method checks the context of the string being represented. If the
        string (`data`) is determined to be a key within a YAML mapping, it
        delegates to the parent class's `represent_str` method, which typically
        results in unquoted keys if they are simple scalars. For all other
        strings (i.e., values, or strings where the key context cannot be
        reliably determined), it forces the string to be represented as a
        double-quoted scalar (`style='"'`). This ensures that string values are
        explicitly quoted, enhancing clarity, while keys can remain unquoted
        for better readability.

        Args:
            data (str): The string data to be represented in YAML.

        Returns:
            yaml.Node: The YAML node representing the string, with appropriate quoting.
        """
        # Check if we're inside a key context by inspecting the parent node
        if self.context_stack and isinstance(self.context_stack[-1], yaml.MappingNode):
            is_key = (self.context_stack[-1].value.index(self.cur_node) % 2 == 0)
            if is_key:
                return super().represent_str(data)  # leave keys unquoted if possible
        return self.represent_scalar('tag:yaml.org,2002:str', data, style='"')

SmartQuotedStringDumper.add_representer(str, SmartQuotedStringDumper.represent_str)

def enforce_scalar_types(obj):
    """
    Recursively convert numeric-looking strings to proper int/float types
    before dumping to YAML, to prevent PyYAML from quoting them.
    """
    if isinstance(obj, dict):
        return {k: enforce_scalar_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [enforce_scalar_types(v) for v in obj]
    elif isinstance(obj, str):
        # Try to convert to int
        if obj.isdigit():
            return int(obj)
        try:
            return float(obj)
        except ValueError:
            return obj
    return obj
