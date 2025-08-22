from typing import Any
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
