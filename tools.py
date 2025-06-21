import re
from typing import List, Dict, Optional, Annotated
from bs4 import BeautifulSoup
import os
import requests
import tiktoken
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_tool = TavilySearchResults(k=6)

@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    def num_tokens_from_string(string: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def clean_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=" ").strip()
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        return text

    results = []
    skipped_urls = []
    token_limit = 100000  # Maximum tokens per page

    for url in urls:
        try:
            response = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
                },
                timeout=10
            )
            response.raise_for_status()
            
            # Clean the text first
            cleaned_text = clean_text(response.text)
            
            # Check token count
            token_count = num_tokens_from_string(cleaned_text)
            
            if token_count > token_limit:
                skipped_urls.append(f"{url} (tokens: {token_count})")
                continue
                
            results.append(
                f'<Document name="{BeautifulSoup(response.text, "html.parser").title.string if BeautifulSoup(response.text, "html.parser").title else url.strip()}">\n{cleaned_text}\n</Document>'
            )
        except Exception as e:
            skipped_urls.append(f"{url} (error: {str(e)})")
            continue

    # Prepare the final output
    output = "\n\n".join(results)
    if skipped_urls:
        output += "\n\nSkipped URLs (too large or error):\n" + "\n".join(skipped_urls)
    
    return output

@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
    session_folder: str = None
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline."""
    folder = session_folder if session_folder else "research_sessions/final"
    os.makedirs(folder, exist_ok=True)
    
    with open(os.path.join(folder, file_name), "w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"

@tool
def read_document(
    file_name: Annotated[str, "File path to read the document."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
    session_folder: str = None
) -> str:
    """Read the specified document."""
    folder = session_folder if session_folder else "research_sessions/final"
    with open(os.path.join(folder, file_name), "r") as file:
        lines = file.readlines()
    if start is not None:
        start = 0
    return "\n".join(lines[start:end])

@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
    session_folder: str = None
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    folder = session_folder if session_folder else "research_sessions/final"
    os.makedirs(folder, exist_ok=True)
    
    with open(os.path.join(folder, file_name), "w") as file:
        file.write(content)
    return f"Document saved to {file_name}"

@tool
def edit_document(
    file_name: Annotated[str, "File path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
    session_folder: str = None
) -> Annotated[str, "File path of the edited document."]:
    """Edit a document by inserting text at specific line numbers."""
    folder = session_folder if session_folder else "research_sessions/final"
    file_path = os.path.join(folder, file_name)
    
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Process insertions in order
    sorted_inserts = sorted(inserts.items())

    # Insert text at specified line numbers
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    # Save edited document to file
    with open(file_path, "w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}" 