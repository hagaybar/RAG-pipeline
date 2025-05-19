import os
import re
import pandas as pd
from bs4 import BeautifulSoup, Tag, NavigableString
from typing import List, Dict, Tuple, Set, Optional
from scripts.api_clients.openai.gptApiClient import APIClient
import hashlib
from urllib.parse import urlparse
from sklearn.decomposition import PCA
import numpy as np
import logging

# Initialize logger
logger = logging.getLogger(__name__)  # Logger name is the script name
logger.setLevel(logging.DEBUG)  # Set logging level

# Create file handler
file_handler = logging.FileHandler("web_processor.log")  # Log file
file_handler.setLevel(logging.DEBUG)  # Log level for file

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Log level for console

# Define log format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)



class ContextPreservingProcessor:
    """
    Enhanced processor for academic library websites that preserves contextual relationships.
    Focuses on maintaining semantic connections between related content.
    """
    
    def __init__(self):
        # Common patterns for navigation/boilerplate content in library websites
        self.nav_patterns = [
            r'facebook', r'twitter', r'instagram', r'youtube',
            r'copyright', r'terms of use', r'privacy policy',
            r'login', r'sign in', r'my account'
        ]
        
        # Elements likely to contain navigation/boilerplate
        self.nav_elements = ['nav', 'header', 'footer']
        self.nav_classes = ['menu', 'navigation', 'navbar', 'header', 'footer', 'sidebar']
        self.nav_ids = ['nav', 'menu', 'header', 'footer', 'sidebar']
        
        # Heading elements
        self.heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        
        # Content elements
        self.content_tags = ['p', 'div', 'section', 'article', 'main', 'ul', 'ol', 'table']
        
        logger.info("Initialized ContextPreservingProcessor")
        
        self.api_client = APIClient()  # Initialize API client
        logger.info("Initialized API client")


    def process_html_files(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Process multiple HTML files with context preservation.
        
        Args:
            file_paths: List of paths to HTML files
            
        Returns:
            DataFrame with processed content
        """
        all_data = []



        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Extract filename as page identifier
                page_name = os.path.basename(file_path)
                
                # Process the file
                file_data = self.process_single_html(html_content, page_name, file_path)
                all_data.extend(file_data)
                
                logger.info(f"Processed {page_name}: extracted {len(file_data)} content blocks")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Convert to DataFrame
        if not all_data:
            logger.warning("No valid data extracted.")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_data)
        
        # Filter and deduplicate
        df = self._deduplicate_content(df)
        # 🔹 Step 1: Prepare formatted text for embeddings
        df = self.prepare_for_embeddings(df)

        # 🔹 Step 2: Generate embeddings using the prepared text
        # try:
        #     df["embedding"] = df["embedding_text"].apply(
        #         lambda text: self.api_client.get_embedding(text) if isinstance(text, str) else None
        #     )
        # except Exception as e:
        #     logger.error(f"Error generating embeddings: {e}")

        return df
    
    def process_single_html(self, html_content: str, page_name: str, source_url: str) -> List[Dict]:
        """
        Process a single HTML document with context preservation.
        
        Args:
            html_content: HTML content as string
            page_name: Identifier for the page
            source_url: Original URL or file path
            
        Returns:
            List of dictionaries containing context-aware content blocks
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script, style elements, and hidden elements
        for element in soup(['script', 'style', 'iframe', 'noscript']):
            element.decompose()
            
        # First pass: remove navigation elements
        # try:
        #     self._remove_navigation_elements(soup)

        # except Exception as e:
        #     logger.error(f"Error in _remove_navigation_elements for {page_name}: {e}")
        
        # Process in different ways to capture context
        content_blocks = []
        
        try:
            self._process_semantic_sections(soup, content_blocks, page_name, source_url) # 1. Process content within clear contextual sections
            self._process_heading_blocks(soup, content_blocks, page_name, source_url) # 2. Process heading-based context blocks
            self._process_contextual_lists(soup, content_blocks, page_name, source_url) # 3. Process lists with their context
            self._process_standalone_paragraphs(soup, content_blocks, page_name, source_url) # 4. Process any remaining important paragraphs without context
            self._process_tables(soup, content_blocks, page_name, source_url)
        except Exception as e:
            logger.error(f"Error in processing for {page_name}: {e}")       
        return content_blocks
    # the method _remove_navigation_elements does not preform well with the current implementation, it is not used.
    def _remove_navigation_elements(self, soup):
        """Remove navigation elements while handling potential NoneType objects."""
        if soup is None:
            logger.warning("Soup object is None, skipping navigation element removal")
            return
            
        try:
            # Use a copy of the elements list to avoid modification during iteration
            elements = list(soup.find_all())
            logger.debug(f"Found {len(elements)} HTML elements to process")
            
            removed_count = 0
            for i, element in enumerate(elements):
                # Skip if element is None
                if element is None:
                    logger.debug(f"Skipping element at index {i}: Element is None")
                    continue
                    
                # Never remove the body or html elements
                if element.name in ['html', 'body', 'head', 'main', 'article', 'section', 'div']:
                    logger.debug(f"Skipping element at index {i}: Essential structure element '{element.name}'")
                    continue
                    
                # Ensure the element is a valid HTML tag
                if not isinstance(element, Tag):
                    logger.debug(f"Skipping element at index {i}: Not a Tag instance (type: {type(element)})")
                    continue

                # Check if element has been decomposed
                if not hasattr(element, 'attrs') or element.attrs is None:
                    logger.debug(f"Skipping element at index {i}: Element attrs is None or missing")
                    continue

                is_nav = False
                reason = ""
                element_classes = []
                element_id = ''

                # Check if element is a navigation tag
                if element.name in self.nav_elements:
                    is_nav = True
                    reason = f"Tag name '{element.name}' is in navigation elements list"
                    logger.debug(f"Element at index {i} identified as navigation: {reason}")

                # Check class attributes
                if not is_nav and 'class' in element.attrs:
                    if isinstance(element.attrs['class'], list):
                        element_classes = element.attrs['class']
                    else:
                        element_classes = [element.attrs['class']]
                        logger.debug(f"Element at index {i} has non-list class attribute: {element.attrs['class']}")
                    
                    class_names = ' '.join(element_classes).lower()
                    for nav in self.nav_classes:
                        if nav in class_names:
                            # Don't mark as navigation if it's a container that might have content
                            if element.name in ['div', 'section', 'article', 'main']:
                                # Only consider it navigation if it has specific navigation classes
                                nav_specific = ['menu', 'navbar', 'nav-', 'navigation', 'header-nav']
                                has_specific_nav = any(specific in class_names for specific in nav_specific)
                                if not has_specific_nav:
                                    logger.debug(f"Element at index {i} has '{nav}' but is a content container, not removing")
                                    continue
                                    
                            is_nav = True
                            reason = f"Class name contains '{nav}'"
                            logger.debug(f"Element at index {i} identified as navigation: {reason}")
                            break
                
                # Check id attributes
                if not is_nav and 'id' in element.attrs:
                    if isinstance(element.attrs['id'], str):
                        element_id = element.attrs['id']
                    else:
                        element_id = str(element.attrs['id'])
                        logger.debug(f"Element at index {i} has non-string id attribute: {element.attrs['id']}")
                    
                    if element_id.lower() in self.nav_ids:
                        is_nav = True
                        reason = f"ID '{element_id}' is in navigation IDs list"
                        logger.debug(f"Element at index {i} identified as navigation: {reason}")

                # Remove navigation elements
                if is_nav:
                    logger.debug(f"Removing navigation element: {element.name} (reason: {reason}) (class: {element_classes}) (id: {element_id})")
                    # Store element info before decomposing it
                    element_name = element.name
                    # Now decompose the element
                    element.decompose()
                    removed_count += 1
                    
                    # Important: continue to next iteration immediately after decomposing
                    continue

            logger.info(f"Successfully processed all elements. Removed {removed_count} navigation elements.")

        except Exception as e:
            logger.error(f"Error in _remove_navigation_elements: {e}", exc_info=True)
            # Additional detailed error information
            logger.debug(f"Error occurred with soup type: {type(soup)}")
            if 'element' in locals():
                logger.debug(f"Last processed element: {element.name if hasattr(element, 'name') else 'unknown'}")
                logger.debug(f"Element attributes: {element.attrs if hasattr(element, 'attrs') else 'no attrs'}")
                logger.debug(f"Element index: {i if 'i' in locals() else 'unknown'}")

    def _process_semantic_sections(self, soup, content_blocks, page_name, source_url):
        """Process content within semantic sections like main, article, section."""
        for section_tag in ['main', 'article', 'section']:
            for section in soup.find_all(section_tag):
                # Get section heading if available
                section_heading = None
                heading = section.find(self.heading_tags)
                if heading:
                    section_heading = heading.get_text(strip=True)
                
                # Process content within the section
                for content_el in section.find_all(['p', 'div', 'ul', 'ol', 'table']):
                    # Skip if it contains mostly navigation patterns
                    if self._is_likely_navigation(content_el):
                        continue
                        
                    # Get meaningful text
                    content_text = content_el.get_text(strip=True)
                    if not content_text or len(content_text) < 15:  # Skip very short content
                        continue
                    
                    # Create a context-aware content block
                    block = {
                        'content_type': 'section_content',
                        'heading': section_heading,
                        'text': content_text,
                        'tag': content_el.name,
                        'source_page': page_name,
                        'source_url': source_url,
                        'importance': 7
                    }
                    
                    content_blocks.append(block)
    
    def _process_heading_blocks(self, soup, content_blocks, page_name, source_url):
        """Process content blocks based on headings."""
        # Find all headings
        headings = soup.find_all(self.heading_tags)
        
        for i, heading in enumerate(headings):
            heading_text = heading.get_text(strip=True)
            if not heading_text:
                continue
                
            # Determine heading level
            level = int(heading.name[1])
            
            # Find content until next heading of same or higher level
            next_heading = None
            for h in headings[i+1:]:
                next_level = int(h.name[1])
                if next_level <= level:
                    next_heading = h
                    break
            
            # Collect content between headings
            contents = []
            current = heading.next_sibling
            
            while current and (not next_heading or current != next_heading):
                if isinstance(current, Tag) and current.name in ['p', 'ul', 'ol', 'div', 'table']:
                    # Skip navigation-like content
                    if not self._is_likely_navigation(current):
                        text = current.get_text(strip=True)
                        if text and len(text) > 10:  # Skip very short content
                            contents.append(text)
                
                # Move to next sibling
                if getattr(current, 'next_sibling', None):
                    current = current.next_sibling
                else:
                    break
            
            # Create content block with heading and all related content
            if contents:
                block = {
                    'content_type': 'heading_block',
                    'heading': heading_text,
                    'text': heading_text + ": " + " ".join(contents),
                    'heading_level': level,
                    'tag': heading.name,
                    'source_page': page_name,
                    'source_url': source_url,
                    'importance': 9 - level  # Higher importance for higher level headings
                }
                
                content_blocks.append(block)
            else:
                # Add just the heading if it has no content
                block = {
                    'content_type': 'standalone_heading',
                    'heading': heading_text,
                    'text': heading_text,
                    'heading_level': level,
                    'tag': heading.name,
                    'source_page': page_name,
                    'source_url': source_url,
                    'importance': 8 - level
                }
                
                content_blocks.append(block)
    
    def _process_contextual_lists(self, soup, content_blocks, page_name, source_url):
        """Process lists with their contextual information."""
        for list_tag in ['ul', 'ol']:
            for list_el in soup.find_all(list_tag):
                # Skip if already processed as part of a section
                if list_el.find_parent(['main', 'article', 'section']):
                    already_processed = False
                    for parent in list_el.parents:
                        if parent.name in ['main', 'article', 'section']:
                            already_processed = True
                            break
                    if already_processed:
                        continue
                
                # Find context for this list
                context = self._find_element_context(list_el)
                list_items = list_el.find_all('li')
                
                # If list has items, create a context block
                if list_items:
                    # Get all items as text
                    items_text = [item.get_text(strip=True) for item in list_items]
                    items_text = [text for text in items_text if text]  # Remove empty
                    
                    if items_text:
                        # Create block with context + items
                        list_text = context + ": " + ", ".join(items_text) if context else ", ".join(items_text)
                        
                        block = {
                            'content_type': 'list_with_context',
                            'heading': context,
                            'text': list_text,
                            'tag': list_tag,
                            'source_page': page_name,
                            'source_url': source_url,
                            'importance': 6,
                            'item_count': len(items_text)
                        }
                        
                        content_blocks.append(block)
    
    def _process_standalone_paragraphs(self, soup, content_blocks, page_name, source_url):
        """Process remaining paragraphs that weren't captured in context blocks."""
        for p in soup.find_all('p'):
            # Skip if already processed as part of a section or heading block
            if p.find_parent(['main', 'article', 'section']):
                already_processed = False
                for parent in p.parents:
                    if parent.name in ['main', 'article', 'section']:
                        already_processed = True
                        break
                if already_processed:
                    continue
            
            # Get text and skip if too short
            text = p.get_text(strip=True)
            if not text or len(text) < 30:  # Higher threshold for standalone paragraphs
                continue
                
            # Skip if navigation-like
            if self._is_likely_navigation(p):
                continue
            
            # Find context for this paragraph
            context = self._find_element_context(p)
            
            # Create block
            block = {
                'content_type': 'standalone_paragraph',
                'heading': context,
                'text': (context + ": " + text) if context else text,
                'tag': 'p',
                'source_page': page_name,
                'source_url': source_url,
                'importance': 5
            }
            
            content_blocks.append(block)
    
    def _find_element_context(self, element):
        """Find the most relevant heading for a given element, prioritizing real headings over inline bolded text."""
        context = None
        highest_priority_heading = None  # Track the best heading

        # Traverse up the DOM tree to find the nearest actual heading tag
        parent = element.parent
        max_depth = 10  # Limit to avoid infinite loops
        while parent is not None and parent.name != '[document]' and max_depth > 0:
            max_depth -= 1
            for heading in parent.find_all_previous(self.heading_tags):  # Only check <h1> to <h6>
                heading_text = heading.get_text(strip=True)
                heading_level = int(heading.name[1])  # Extract level (e.g., "h2" → 2)

                # Prefer higher-level headings (h1/h2) over lower-level ones
                if highest_priority_heading is None or heading_level < highest_priority_heading[1]:
                    highest_priority_heading = (heading_text, heading_level)

            parent = parent.parent  # Move up in hierarchy

        # If a valid heading is found, use it
        if highest_priority_heading:
            context = highest_priority_heading[0]

        return context

    
    def _is_likely_navigation(self, element): # First ensure the element is a Tag; if not, it isn’t navigation. if not isinstance(element, Tag): return False

        # Check classes and IDs for navigation hints
        for attr in ['class', 'id']:
            if element.get(attr):
                attr_value = ' '.join(element[attr]) if isinstance(element[attr], list) else element[attr]
                attr_value = attr_value.lower()
                if any(nav in attr_value for nav in ['nav', 'menu', 'footer', 'header']):
                    return True

        # Check text content for common navigation patterns
        text = element.get_text(strip=True).lower()
        # Short text with navigation keywords is likely navigation
        if len(text) < 100:
            nav_keywords = sum(1 for pattern in self.nav_patterns if re.search(pattern, text, re.IGNORECASE))
            if nav_keywords >= 2:
                return True

        return False

    # def _is_likely_navigation(self, element):
    #     """Check if an element is likely to be navigation or boilerplate."""
    #     if not element:
    #         return False
            
    #     # Check classes and IDs for navigation hints
    #     for attr in ['class', 'id']:
    #         if element.get(attr):
    #             attr_value = ' '.join(element[attr]) if isinstance(element[attr], list) else element[attr]
    #             attr_value = attr_value.lower()
    #             if any(nav in attr_value for nav in ['nav', 'menu', 'footer', 'header']):
    #                 return True
        
    #     # Check text content for common navigation patterns
    #     text = element.get_text(strip=True).lower()
        
    #     # Short text with navigation keywords is likely navigation
    #     if len(text) < 100:
    #         nav_keywords = sum(1 for pattern in self.nav_patterns if re.search(pattern, text, re.IGNORECASE))
    #         if nav_keywords >= 2:
    #             return True
        
    #     return False
    
    def _deduplicate_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter and deduplicate content blocks while preserving context.
        """
        if df.empty:
            return df
        
        # 1. Remove very similar texts (more conservative approach for context preservation)
        text_hashes = {}
        duplicate_indices = []
        
        for idx, row in df.iterrows():
            # Create a normalized version of the text for comparison
            text = row['text'].lower()
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Create hash of the text
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            
            # If hash exists, it's a duplicate
            if text_hash in text_hashes:
                # Compare importance to keep the more important version
                existing_idx = text_hashes[text_hash]
                existing_importance = df.loc[existing_idx, 'importance']
                
                if row['importance'] > existing_importance:
                    # This version is more important, swap
                    duplicate_indices.append(existing_idx)
                    text_hashes[text_hash] = idx
                else:
                    # Existing version is more important or equal
                    duplicate_indices.append(idx)
            else:
                text_hashes[text_hash] = idx
        
        # Remove duplicates
        df_deduped = df[~df.index.isin(duplicate_indices)].copy()
        
        logger.info(f"Removed {len(duplicate_indices)} exact duplicates")
        
        # 2. Handle text containment (more carefully to preserve context)
        # Sort by length (longest first)
        df_deduped = df_deduped.sort_values('text', key=lambda x: x.str.len(), ascending=False)
        contained_indices = []
        for i, row1 in df_deduped.iterrows():
            text1 = row1['text']
            source1 = row1['source_page']
            
            # Skip short text
            if len(text1) < 50:
                continue
                
            # Only compare with other texts from same source
            for j, row2 in df_deduped.iterrows():
                if i == j or j in contained_indices:
                    continue
                    
                text2 = row2['text']
                source2 = row2['source_page']
                
                # Only check containment for same source page
                if source1 != source2:
                    continue
                
                # If second text is mostly contained in first text
                if len(text2) < len(text1) and text2 in text1:
                    # Check if it's a substantial portion
                    if len(text2) / len(text1) > 0.2:
                        contained_indices.append(j)
        
        # Remove contained texts
        df_final = df_deduped[~df_deduped.index.isin(contained_indices)].copy()
        
        logger.info(f"Removed {len(contained_indices)} contained texts")
        
        return df_final
    
    def prepare_for_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare content for embedding by creating a formatted text field.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            DataFrame with embedding-ready text
        """
        if df.empty:
            return df
        
        # Create formatted text for embedding
        df = df.fillna("")
        df['embedding_text'] = df.apply(
            lambda row: f"Page: {row['source_page']} | "
                       f"Type: {row['content_type']} | "
                       f"{('Heading: ' + str(row['heading']) + ' | ') if pd.notna(row['heading']) and row['heading'] else ''}"
                       f"Content: {row['text']}",
            axis=1
        )
        
        return df

    def _process_tables(self, soup, content_blocks, page_name, source_url):
        """Extracts table data while preserving column relationships."""
        tables = soup.find_all("table")

        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")

                if not cells:
                    continue  # Skip empty rows with no cells

                # Extract all available columns
                extracted_values = [cell.get_text(strip=True) for cell in cells]

                # Format text representation of the table row
                if len(extracted_values) == 1:
                    # Handle single-column tables
                    table_text = extracted_values[0]
                else:
                    # Handle multi-column tables, joining columns with a separator
                    table_text = " | ".join(extracted_values)

                # Find context (nearest heading)
                heading = None
                previous = table.find_previous_sibling(self.heading_tags)
                if previous:
                    heading = previous.get_text(strip=True)

                content_blocks.append({
                    "content_type": "table_row",
                    "heading": heading,
                    "text": table_text,
                    "tag": "table",
                    "source_page": page_name,
                    "source_url": source_url,
                    "importance": 8
                })




# Example usage
if __name__ == "__main__":
    import os
    import glob
    
    # 1. Get all HTML files from the directory
    directory = r"C:\Users\hagaybar\OneDrive - Tel-Aviv University\My Personal files\systems\AI Project\test_lib_data_html"
    
    # Use glob to find all HTML files
    html_files = glob.glob(os.path.join(directory, "*.html"))
    html_files.extend(glob.glob(os.path.join(directory, "*.htm")))
    
    logger.info(f"Found {len(html_files)} HTML files to process")
    for file in html_files:
        logger.info(f"  - {os.path.basename(file)}")
    
    # 2. Process HTML files with context preservation
    processor = ContextPreservingProcessor()
    processed_df = processor.process_html_files(html_files)
    
    
    # 3. Save the processed data
    output_file = "library_content_with_context.xlsx"
    processed_df.to_excel(output_file, index=False)
    logger.info(f"\nSaved context-preserving processed data to {output_file}")
    
    # 4. Sample query
    query = "When is the library closed for holidays?"
    logger.info(f"\nSample query: '{query}'")
    logger.info("In a real implementation, this would retrieve the most similar content based on embeddings")


