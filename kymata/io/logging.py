log_message = "%(asctime)s | %(levelname).2s | %(filename)s:%(lineno)d | \t%(message)s"
date_format = "%Y-%m-%d %H:%M:%S"


def describe_higher_lower_than(val: float, reference: float = 0, include_preposition: bool = True) -> str:
    """
    Compares `val` to `reference` and returns:

    - "higher than" if val is higher than reference
    - "lower than" if val is lower than reference
    - "the same as" if val is the same as reference

    prepositions ("than", "as") and their preceding spaces are omitted if include_preposition is set to False.
    """
    if val > reference:
        comparative_adjective = "higher"
        preposition = "than"
    elif val < reference:
        comparative_adjective = "lower"
        preposition = "than"
    else:
        comparative_adjective = "the same"
        preposition = "as"

    if include_preposition:
        return f"{comparative_adjective} {preposition}"
    else:
        return comparative_adjective
