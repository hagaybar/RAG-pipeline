import os
import re
from typing import Callable, List, Optional, Union
from bs4 import BeautifulSoup
import pandas as pd

class DataProcessor:
    """A class for handling file operations and text processing for the RAG system."""

    def __init__(self, encoding: str = "utf-8"):
        """Initialize the DataProcessor with configurable file encoding."""
        self.encoding = encoding

    def read_file(self, file_path: str) -> str:
        """
        Read a file and return its contents as a string.

        Args:
            file_path (str): Path to the file to be read.

        Returns:
            str: The contents of the file as a string, normalized for bidirectional text.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If there is an issue reading the file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at {file_path} does not exist.")
        
        try:
            with open(file_path, "r", encoding=self.encoding) as file:
                content = file.read()
            # Optional: Normalize bidirectional text (if needed)
            return content
        except UnicodeDecodeError:
            raise IOError(f"Failed to decode the file using {self.encoding}. Ensure the encoding matches the file.")
        except Exception as e:
            raise IOError(f"An error occurred while reading the file: {e}")
   
    def parse_html_to_dataframe(self, html_content: str) -> pd.DataFrame:
        """
        Parse HTML content and extract data into a pandas DataFrame.

        Args:
            html_content (str): The HTML content as a string.

        Returns:
            pd.DataFrame: A DataFrame containing parsed data.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        parsed_rows = []

        for table_index, table in enumerate(tables):
            for row_index, row in enumerate(table.find_all('tr')):
                cells = row.find_all(['td', 'th'])
                row_data = {
                    'table_index': table_index,
                    'row_index': row_index,
                    'content_type': 'table_row',
                }

                # Extract each cell's content into separate columns
                for cell_index, cell in enumerate(cells):
                    row_data[f'cell_{cell_index}'] = cell.get_text(strip=True)

                parsed_rows.append(row_data)

        return pd.DataFrame(parsed_rows)

    def clean_text(self, text: str) -> str:
        """
        Perform basic text cleaning, including removing excessive whitespace and special characters.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'[\r\n\t]', '', text)  # Remove newline and tab characters
        return text.strip()

    def extract_metadata(self, html_content: str) -> List[dict]:
        """
        Extract metadata from the HTML content, such as table or list indices.

        Args:
            html_content (str): The HTML content as a string.

        Returns:
            List[dict]: A list of metadata dictionaries.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        metadata = []

        # Extract table metadata
        tables = soup.find_all('table')
        for table_index, table in enumerate(tables):
            metadata.append({
                'table_index': table_index,
                'rows': len(table.find_all('tr')),
            })

        # Extract list metadata
        lists = soup.find_all(['ul', 'ol'])
        for list_index, list_tag in enumerate(lists):
            metadata.append({
                'list_index': list_index,
                'items': len(list_tag.find_all('li')),
            })

        return metadata

    def save_dataframe_to_csv(self, dataframe: pd.DataFrame, output_path: str):
        """
        Save a pandas DataFrame to a CSV file.

        Args:
            dataframe (pd.DataFrame): The DataFrame to save.
            output_path (str): The path to the output CSV file.
        """
        dataframe.to_csv(output_path, index=False, encoding=self.encoding)

    def process_html_file(self, file_path: str, tags: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Process an HTML file and convert it into a pandas DataFrame by extracting specified elements.

        Args:
            file_path (str): The path to the HTML file.
            tags (List[str], optional): The list of HTML tags to extract. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing parsed and cleaned data with metadata.
        """
        html_content = self.read_file(file_path)
        df = self.parse_html_elements_to_dataframe(html_content, tags=tags)
        df = df.applymap(lambda x: self.clean_text(x) if isinstance(x, str) else x)
        return df

    def process_directory(directory: str) -> pd.DataFrame:
        """
        Process all HTML files in a directory and combine parsed content.

        Args:
            directory (str): Path to the directory containing HTML files.

        Returns:
            pd.DataFrame: Combined DataFrame for all files.
        """
        all_data = []
        processor = DataProcessor()

        for file_name in os.listdir(directory):
            if file_name.endswith(".html"):
                with open(os.path.join(directory, file_name), "r", encoding="utf-8") as file:
                    html_content = file.read()
                    df = processor.parse_html(html_content, file_name)
                    all_data.append(df)

        return pd.concat(all_data, ignore_index=True)
 
    def export_df_to_excel(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save a DataFrame to an Excel file.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            output_path (str): The file path to save the Excel file.
        """
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"DataFrame saved to {output_path}")

    def parse_html_elements_to_dataframe(
        self, html_content: str, tags: Optional[List[str]] = None, include_nested: bool = True
    ) -> pd.DataFrame:
        """
        Parse specified HTML elements and extract their text and metadata into a DataFrame.

        Args:
            html_content (str): The HTML content as a string.
            tags (List[str], optional): List of tag names to extract. Defaults to a set of common elements.
            include_nested (bool): If True, calculates the nesting level for each element.

        Returns:
            pd.DataFrame: DataFrame containing extracted text and metadata.
        """
        if tags is None:
            # Default set: tables, divs, spans, paragraphs, headers, and list items.
            tags = ['table', 'div', 'span', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li']

        soup = BeautifulSoup(html_content, 'html.parser')
        data = []
        # Loop over each tag type
        for tag in tags:
            for element in soup.find_all(tag):
                # Skip empty elements if desired
                element_text = element.get_text(strip=True)
                if not element_text:
                    continue
                nesting_level = self.get_nesting_level(element) if include_nested else None
                data.append({
                    'tag': tag,
                    'text': self.clean_text(element_text),
                    'attributes': element.attrs,
                    'nesting_level': nesting_level,
                })
        return pd.DataFrame(data)

    def get_nesting_level(self, element: BeautifulSoup) -> int:
        """
        Calculate the nesting level of an element.
        The document root is level 0.
        """
        level = 0
        parent = element.parent
        while parent is not None and parent.name != '[document]':
            level += 1
            parent = parent.parent
        return level




# Usage:
if __name__ == "__main__":
    processor = DataProcessor()
    html_file = "C:\\Users\\hagaybar\\OneDrive - Tel-Aviv University\\My Personal files\\systems\\AI Project\\codebase\\scripts\\data_processing\\Opening.html"

    file_content = processor.read_file(html_file)
    # print("Raw file content (first 500 chars):", file_content[:500])
    # cleaned_content = processor.clean_text(file_content) 
    # print("Cleaned content (first 500 chars):", cleaned_content[:500])
    df = processor.process_html_file(html_file) 
    print("Extracted DataFrame:") 
    print(df.head(10))


    # steps = [
    #                     processor.strip_html,
    #                     # processor.remove_urls,
    #                     # processor.lowercase,
    #                     # processor.normalize_whitespace

    #         ]
    # def save_text_to_file(cleaned_text):
    #     with open("cleaned_text.txt", "w", encoding="utf-8") as f:
    #         f.write(cleaned_text)
    # cleaned_text = processor.clean_text(file_content, steps)
    # save_text_to_file(cleaned_text)
    # chunks = processor.chunk_text(cleaned_text, 500)
    # print(stripped)


        # def parse_html(self, html_content: str, file_name: str) -> pd.DataFrame:
    #     """
    #     Parse HTML content into a DataFrame with content type and metadata.

    #     Args:
    #         html_content (str): HTML content to parse.
    #         file_name (str): Name of the source file.

    #     Returns:
    #         pd.DataFrame: DataFrame containing parsed content with metadata.
    #     """
    #     soup = BeautifulSoup(html_content, "html.parser")
    #     data = []

    #     # Parse tables
    #     for table_idx, table in enumerate(soup.find_all("table")):
    #         for row_idx, row in enumerate(table.find_all("tr")):
    #             cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
    #             data.append({
    #                 "content": " | ".join(cells),
    #                 "content_type": "table_row",
    #                 "file_name": file_name,
    #                 "metadata": {"table_index": table_idx + 1, "row": row_idx + 1}
    #             })

    #     # Parse unordered lists (ul/li)
    #     for ul_idx, ul in enumerate(soup.find_all("ul")):
    #         for li_idx, li in enumerate(ul.find_all("li")):
    #             marker = li.find("span") or li.find("i")
    #             bullet_marker = marker.get_text(strip=True) if marker else ""
    #             text = li.get_text(strip=True)
    #             data.append({
    #                 "content": f"{bullet_marker} {text}".strip(),
    #                 "content_type": "bullet_list",
    #                 "file_name": file_name,
    #                 "metadata": {"list_index": ul_idx + 1, "item_index": li_idx + 1}
    #             })

    #     # Parse headers (h1-h6)
    #     for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
    #         level = int(header.name[1])  # Extract the level (e.g., 1 for h1)
    #         data.append({
    #             "content": header.get_text(strip=True),
    #             "content_type": "header",
    #             "file_name": file_name,
    #             "metadata": {"level": level}
    #         })

    #     # Parse plain text paragraphs
    #     for paragraph in soup.find_all("p"):
    #         data.append({
    #             "content": paragraph.get_text(strip=True),
    #             "content_type": "plain_text",
    #             "file_name": file_name,
    #             "metadata": {}
    #         })

    #     # Fallback for unhandled elements
    #     handled_tags = {"table", "ul", "h1", "h2", "h3", "h4", "h5", "h6", "p"}
    #     for element in soup.find_all():
    #         if element.name not in handled_tags and element.get_text(strip=True):
    #             data.append({
    #                 "content": element.get_text(strip=True),
    #                 "content_type": "unhandled_element",
    #                 "file_name": file_name,
    #                 "metadata": {"tag": element.name, "attributes": element.attrs}
    #             })

    #     # Return as DataFrame
    #     return pd.DataFrame(data)
