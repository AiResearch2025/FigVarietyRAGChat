from enum import Enum, auto

class QueryCase(Enum):
    """
    Represents the different cases for user queries about fig varieties.
    """

    # Case A: User knows the variety
    KNOWN_VARIETY_EXISTS = auto()  # Variety exists and the query is clear
    KNOWN_VARIETY_TYPO = auto()  # Variety name has a typo
    KNOWN_VARIETY_NEEDS_CLARIFICATION = auto()  # Query about a known variety is vague

    # Case B: User does not know the variety
    UNKNOWN_VARIETY_GENERAL_QUERY = auto()  # General query without a variety name
    UNKNOWN_VARIETY_IMAGE_QUERY = auto()  # Query with an image to identify the variety

def classify_query(query: str) -> QueryCase:
    """
    Classifies a user query into one of the QueryCase categories.

    This is a placeholder implementation. The actual classification logic
    will require more sophisticated NLP techniques.
    """
    # Simple keyword-based classification for demonstration
    query_lower = query.lower()

    # This is a very basic example and would need to be expanded
    # with more robust logic, potentially using an LLM or fuzzy string matching.

    if "추천" in query_lower or "어떤" in query_lower and "품종" in query_lower:
        return QueryCase.UNKNOWN_VARIETY_GENERAL_QUERY

    # This is a placeholder for image query detection
    if "사진" in query_lower or "이미지" in query_lower:
        return QueryCase.UNKNOWN_VARIETY_IMAGE_QUERY

    # In a real implementation, you would check against your list of varieties.
    # For now, we'll assume if it doesn't fit the above, it's a known variety query.
    # Further logic would be needed to detect typos or vague questions.

    # Placeholder for KNOWN_VARIETY_TYPO
    # e.g., if fuzzy_match(query_variety, known_varieties) > 0.8 and not in known_varieties:
    #     return QueryCase.KNOWN_VARIETY_TYPO

    # Placeholder for KNOWN_VARIETY_NEEDS_CLARIFICATION
    if "맛있" in query_lower and "어떤" in query_lower:
        return QueryCase.KNOWN_VARIETY_NEEDS_CLARIFICATION

    return QueryCase.KNOWN_VARIETY_EXISTS

if __name__ == '__main__':
    # Example Usage
    queries = [
        "BNR은 당도가 높나요?",
        "Ciccio Vero에 대해 알려줘",
        "브런즈윅은 어떤 맛인가요?",
        "달콤한 무화과 품종 추천해줘",
        "이 사진 속 무화과는 무슨 품종이야?",
    ]

    for q in queries:
        case = classify_query(q)
        print(f"Query: '{q}'\nCase: {case.name}\n")
