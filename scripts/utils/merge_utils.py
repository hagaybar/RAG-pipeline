"""
This module contains the `deep_merge` utility function, which performs a
recursive merge of two dictionaries. During the merge, values from the
second dictionary (overrides) take precedence over the first, and this
logic is applied deeply to any nested dictionaries.
"""
def deep_merge(base: dict, overrides: dict) -> dict:
    """
    Recursively merges two dictionaries, `base` and `overrides`.

    If a key exists in both dictionaries and both values are dictionaries,
    the function will recursively merge these nested dictionaries. Otherwise,
    the value from the `overrides` dictionary takes precedence.

    Args:
        base (dict): The base dictionary.
        overrides (dict): The dictionary containing values to override `base`.

    Returns:
        dict: A new dictionary representing the merged result.
    """
    result = base.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
