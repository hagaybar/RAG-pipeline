import pathlib
from pathlib import Path
import google.generativeai as genai
import requests
import os
import time
from google.api_core import exceptions

# Initialize API and model outside of functions
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

def create_file_name():
    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    file_name = f"chat_{timestamp}"
    return file_name

def show_file(file_data):
    print("show_file: Starting")
    mime_type = file_data["mime_type"]
    name = None
    file_content = None

    if url := file_data.get("url", None):
        print(f"show_file: Processing URL: {url}")
        name = url
        try:
            response = requests.get(url)
            response.raise_for_status()
            file_content = response.text
        except requests.exceptions.RequestException as e:
           print(f"show_file: Error fetching URL: {e}")
           print("show_file: Returning None")
           return None

    elif data := file_data.get("inline_data", None):
        print("show_file: Processing inline data")
        name = None
        try:
           file_content = data.decode('utf-8', errors='ignore')
        except UnicodeDecodeError as e:
            print(f"show_file: Error decoding data {e}")
            print("show_file: Returning None")
            return None
    elif name := file_data.get("filename", None):
        print(f"show_file: Processing filename: {name}")
        path = Path(name)
        if not path.exists():
            print(f"show_file: Error: local file: {path} does not exist.")
            print("show_file: Returning None")
            return None
        try:
            with open(path, "r") as f:
                file_content = f.read()
        except UnicodeDecodeError:
            try:
                with open(path, "r", encoding="ISO-8859-1") as f:
                    file_content = f.read()
            except Exception as e:
              print(f"show_file: Error reading file {path}, {e}")
              print("show_file: Returning None")
              return None
    else:
        print("show_file: Error: Invalid file data")
        raise ValueError("Either `url` or `inline_data` must be provided.")

    if file_content:
        print("show_file: Returning file content")
        print(f"show_file: Content: {file_content[:100]}...")
        print("show_file: Returning")
        return (name, file_content)
    else:
        print("show_file: Returning None")
        return None

def read_local_file(file_path):
    print(f"read_local_file: Starting with file_path: {file_path}")
    path = Path(file_path)
    if not path.exists():
        print(f"read_local_file: File not found: {path}")
        return

    if path.suffix != '.xml':
        print("read_local_file: Please select an XML file")
        return

    file_data = {
        "mime_type": "text/xml",
        "filename": str(path)
    }

    result = show_file(file_data)
    if result:
        name, content = result
        print(f"read_local_file: Successfully read file: {name}")
        user_question = '''
        You find yourself in a bustling archive, surrounded by rows of bibliographic records detailing centuries of human thought. 
        Each record contains essential metadata—title, author, publication date, publisher, subjects, ISBN, and various identifiers such as VIAF, ISNI, ORCID, and DOI.
        Your mission is to illuminate the relationships hidden in this array of metadata. 
        Craft a compelling narrative that not only describes each record concisely and informatively but also highlights connections—between authors and their influences, between publishers and the movements they helped shape, and between topics, places, and significant historical events. As you wander through the archive, consider the rich context each record holds: Weave together new and fascinating details from linked data sources such as VIAF, Wikidata, DBpedia, or the Library of Congress Name Authority Files. Show how a single author’s VIAF record can open up a global network of collaborations, or how a city listed in Wikidata can shed light on a location’s cultural history. Reveal lesser-known threads linking each title to broader literary, scientific, or cultural movements. Perhaps a single book’s ISBN leads you down a path of related editions spanning continents, or an author’s ORCID number unravels a legacy of academic collaborations.
        Contextualize the subjects by situating them within the broader tapestry of history and scholarship. 
        How did a particular publication date coincide with pivotal world events, and how did these events influence (or were influenced by) the author or the subject matter? 
        By the end of this journey, your readers should see these bibliographic records as intricately interwoven signposts rather than isolated listings.
        Each entry in this narrative should be a gateway to new discoveries, thanks to the rich interplay of metadata and linked data. 
        Above all, bring these records to life—not merely as static references, but as vibrant, interconnected moments in humanity’s collective knowledge.
        '''

        # user_question = input("Please enter a question:\n")
        print(f"read_local_file: User question: {user_question}")

        prompt = f"""
            You are an expert in analyzing data from XML documents. 
            Your task is to extract specific information from the XML content and answer the user's question.

            XML Document Content:
            ```xml
            {content}
            ```

            User's question: {user_question}

            Please provide your answer in a clear and concise manner. If the question is about the number of records,
            please provide the number. If the question is about the names of authors, please provide a list of the authors.
            If the question is about something else, please provide the answer in a clear and concise manner.
        """
        print(f"read_local_file: Prompt: {prompt[:200]}...")
        try:
             ai_response = get_ai_response(prompt)
             if ai_response:
                 print("read_local_file: AI Response:\n", ai_response, "\n")
                 
                 # Save the chat to a file
                 file_name = create_file_name()
                 chat = f"User Question:\n{user_question}\n\nContent Name:\n{name}\n\nAI Response:\n{ai_response}"
                 save_chat_to_file(file_name, chat)
             else:
                 print("read_local_file: AI response is empty")
        except Exception as e:
           print(f"read_local_file: Error getting AI response: {e}")
    else:
        print("read_local_file: Failed to read file")
    print("read_local_file: Returning")

def save_chat_to_file(file_name, chat):
    print(f"save_chat_to_file: Saving chat to file: {file_name}")
    print(f"save_chat_to_file: Chat content:\n{chat}")
    with open(f"{file_name}.txt", "w", encoding="utf-8") as file:
        file.write(chat)
    print(f"save_chat_to_file: Chat saved to {file_name}.txt")

def get_ai_response(prompt):
    print("get_ai_response: Starting")
    try:
        time.sleep(10)  # Add a 1 second delay between API calls
        response = model.generate_content(prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                max_output_tokens=2000,
            ))
        print("get_ai_response: Returning response")
        return response.text
    except exceptions.ResourceExhausted as e:
        print(f"get_ai_response: Error calling ai model: Quota exceeded {e}")
        return None
    except Exception as e:
        print(f"get_ai_response: Error calling ai model: {e}")
        return None

# save_chat_to_file
if __name__ == "__main__":
    file_name_and_path = Path('C:/Users/hagaybar/OneDrive - Tel-Aviv University/My Personal files/systems/AI Project/DATA/BIBS/test_set_2024122216_32904224830004146_new.xml')
    read_local_file(file_name_and_path)