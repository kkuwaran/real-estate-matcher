from typing import Dict, Any
from datetime import date
from IPython.display import display


def show_section(
    title: str, 
    content: Any, 
    use_display: bool = False, 
    width: int = 10
) -> None:
    """Print or display a formatted section with a title and content."""

    header = f"{'=' * width} {title} {'=' * width}"
    print(header)

    if use_display:
        display(content)
    else:
        print(content)

    print()  # extra newline


def serialize_dates(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert any datetime.date objects in the dictionary to ISO-formatted strings."""

    serialized = {}
    for key, value in data.items():
        if isinstance(value, date):
            serialized[key] = value.isoformat()  # "YYYY-MM-DD"
        else:
            serialized[key] = value
    return serialized
