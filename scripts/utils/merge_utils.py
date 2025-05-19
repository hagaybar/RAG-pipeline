def deep_merge(base: dict, overrides: dict) -> dict:
    """
    Recursively merge two dictionaries. `overrides` values take precedence.
    """
    result = base.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
