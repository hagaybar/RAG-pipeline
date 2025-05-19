import google.generativeai as genai
import os
import time
from google.api_core import exceptions

# Initialize API and model outside of functions
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

def get_ai_response(prompt):
    print("get_ai_response: Starting")
    try:
        time.sleep(10)  # Add a 10 second delay between API calls
        response = model.generate_content(prompt)
        print("get_ai_response: Returning response")
        return response.text
    except exceptions.ResourceExhausted as e:
        print(f"get_ai_response: Error calling ai model: Quota exceeded {e}")
        return None
    except Exception as e:
        print(f"get_ai_response: Error calling ai model: {e}")
        return None

if __name__ == "__main__":
    xml_string = """
<?xml version="1.0" encoding="UTF-8"?>
<collection>
	<record>
		<leader>03653nam a2200433 a 4500</leader>
		<controlfield tag="005">20230519003326.0</controlfield>
		<controlfield tag="008">030627m19531974enk     r     000 0 eng d</controlfield>
		<controlfield tag="001">990009882160204146</controlfield>
		<datafield tag="020" ind1=" " ind2=" ">
			<subfield code="a">9780099426776(pbk.)</subfield>
		</datafield>
		<datafield tag="020" ind1=" " ind2=" ">
			<subfield code="a">0099426773(pbk.)</subfield>
		</datafield>
		<datafield tag="035" ind1=" " ind2=" ">
			<subfield code="a">(Aleph)000988216TAU01</subfield>
		</datafield>
		<datafield tag="035" ind1=" " ind2=" ">
			<subfield code="a">(OCoLC)965512</subfield>
		</datafield>
		<datafield tag="041" ind1="0" ind2=" ">
			<subfield code="a">eng</subfield>
		</datafield>
		<datafield tag="090" ind1=" " ind2=" ">
			<subfield code="a">150.1952</subfield>
		</datafield>
		<datafield tag="100" ind1="1" ind2=" ">
			<subfield code="a">Freud, Sigmund,</subfield>
			<subfield code="d">1856-1939</subfield>
			<subfield code="9">lat</subfield>
			<subfield code="8">PreferredLanguageHeading</subfield>
			<subfield code="0">https://open-eu.hosted.exlibrisgroup.com/alma/972NNL_INST/authorities/987007261471505171.jsonld</subfield>
			<subfield code="3">NLI-987007261471505171</subfield>
		</datafield>
		<datafield tag="245" ind1="1" ind2="4">
			<subfield code="a">The standard edition of the complete psychological works of Sigmund Freud /</subfield>
			<subfield code="c">translated from the German under the General editorship of James Strachey : in collaboration with Anna Freud : assisted by Alix Strachey and Alan Tyson : editorial assistant: Angela Richards</subfield>
		</datafield>
		<datafield tag="246" ind1="3" ind2="3">
			<subfield code="a">The complete psychological works of Sigmund Freud</subfield>
		</datafield>
		<datafield tag="260" ind1=" " ind2=" ">
			<subfield code="a">London :</subfield>
			<subfield code="b">Hogarth Press and the Institute of Psycho-Analysis,</subfield>
			<subfield code="c">1953-1974</subfield>
		</datafield>
		<datafield tag="300" ind1=" " ind2=" ">
			<subfield code="a">24 v. :</subfield>
			<subfield code="b">ill.</subfield>
		</datafield>
		<datafield tag="500" ind1=" " ind2=" ">
			<subfield code="a">Title on spine: The complete psychological works of Sigmund Freud</subfield>
		</datafield>
		<datafield tag="500" ind1=" " ind2=" ">
			<subfield code="a">Vol. 1 published 1966</subfield>
		</datafield>
		<datafield tag="500" ind1=" " ind2=" ">
			<subfield code="a">Vols. 4 and 5 reprinted with corrections</subfield>
		</datafield>
		<datafield tag="500" ind1=" " ind2=" ">
			<subfield code="a">Vol. 22, published: London : Vintage : The Hogart Press : The Institute of Psycho-Analysis, [2001] (9780099426776(pbk.))</subfield>
		</datafield>
		<datafield tag="505" ind1="0" ind2=" ">
			<subfield code="a">Contents: v.1. 1886-1899: Pre-Psycho-Analytic publications and unpublished drafts -- v.2. 1893-1895: studies on hysteria / by Josef Breuer and Sigmund Freud -- v.3. 1893-1899: Early psycho-analytic publications -- v.4. 1900: the interpretation of dreams, 1st pt. -- v.5. 1900-1901: the interpretation of dreams, 2nd pt. and, on dreams -- v.6. 1901: The psychopathology of everyday life -- v.7. 1901- 1905: a case of hysteria, three essays on sexuality, and other works --</subfield>
		</datafield>
		<datafield tag="505" ind1="0" ind2=" ">
			<subfield code="a">v.8. 1905: Jokes and their relation to the unconscious -- v.9. 1906-1908: Jensen's 'gradiva' and other works -- v.10. 1909: Two case histories (little Hans' and the 'Rat man') -- v.11. 1910: five lectures on psycho-analysis, Leonardo da Vinci and other works -- v.12. 1911-1913: The case of Schreber, papers on technique, and other works -- v.13. 1913-1914: totem and taboo, and other works -- v.14. 1914-1916: On the history of the psycho-analytic movement, papers on metapsychology, and other works -- v.15. 1915-1916: introductory lectures on psycho-analysis, pts. 1 and 2 --</subfield>
		</datafield>
		<datafield tag="505" ind1="0" ind2=" ">
			<subfield code="a">v.16. 1916-1917: Introductory lectures on psycho-analysis (pt. 3) -- v.17. 1917-1919: An infantile neurosis and other works -- V.18. 1920-1922: Beyond the pleasure principle, Group psychology, and other works -- v.19. 1923-1925: the ego and the ID, and other works -- v.20. 1925-1926: An autobiographical study, inhibitions, symptoms and anxiety, the question of lay analysis, and other works -- v.21. 1927-1931: The future of An illusion, civilization and its discontents and other works -- v.22. 1932-36: new introductory lectures on psycho-analysis and other works -- v.23. 1937-1939: Moses and monotheism, An outline of psycho- analysis, and other works -- v.24. indexes and bibliographies / compiled by Angela Richards</subfield>
		</datafield>
		<datafield tag="590" ind1=" " ind2=" ">
			<subfield code="a">EL/YL</subfield>
		</datafield>
		<datafield tag="650" ind1=" " ind2="7">
			<subfield code="a">Psychoanalysis</subfield>
			<subfield code="9">lat</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="0">https://open-eu.hosted.exlibrisgroup.com/alma/972NNL_INST/authorities/987007546289205171.jsonld</subfield>
			<subfield code="3">NLI-987007546289205171</subfield>
		</datafield>
		<datafield tag="700" ind1="1" ind2=" ">
			<subfield code="a">Strachey, James</subfield>
			<subfield code="9">lat</subfield>
			<subfield code="8">PreferredLanguageHeading</subfield>
			<subfield code="0">https://open-eu.hosted.exlibrisgroup.com/alma/972NNL_INST/authorities/987007268543505171.jsonld</subfield>
			<subfield code="3">NLI-987007268543505171</subfield>
		</datafield>
		<datafield tag="700" ind1="1" ind2=" ">
			<subfield code="a">Freud, Anna,</subfield>
			<subfield code="d">1895-1982</subfield>
			<subfield code="9">lat</subfield>
			<subfield code="8">PreferredLanguageHeading</subfield>
			<subfield code="0">https://open-eu.hosted.exlibrisgroup.com/alma/972NNL_INST/authorities/987007261472805171.jsonld</subfield>
			<subfield code="3">NLI-987007261472805171</subfield>
		</datafield>
		<datafield tag="700" ind1="1" ind2=" ">
			<subfield code="a">Strachey, Alix,</subfield>
			<subfield code="d">1892-1973</subfield>
			<subfield code="9">lat</subfield>
			<subfield code="8">PreferredLanguageHeading</subfield>
			<subfield code="0">https://open-eu.hosted.exlibrisgroup.com/alma/972NNL_INST/authorities/987007280876605171.jsonld</subfield>
			<subfield code="3">NLI-987007280876605171</subfield>
		</datafield>
		<datafield tag="700" ind1="1" ind2=" ">
			<subfield code="a">Tyson, Alan</subfield>
			<subfield code="9">lat</subfield>
			<subfield code="8">PreferredLanguageHeading</subfield>
			<subfield code="0">https://open-eu.hosted.exlibrisgroup.com/alma/972NNL_INST/authorities/987007463255805171.jsonld</subfield>
			<subfield code="3">NLI-987007463255805171</subfield>
		</datafield>
		<datafield tag="700" ind1="1" ind2=" ">
			<subfield code="a">Breuer, Josef,</subfield>
			<subfield code="d">1842-1925</subfield>
			<subfield code="9">lat</subfield>
			<subfield code="8">PreferredLanguageHeading</subfield>
			<subfield code="0">https://open-eu.hosted.exlibrisgroup.com/alma/972NNL_INST/authorities/987007259053805171.jsonld</subfield>
			<subfield code="3">NLI-987007259053805171</subfield>
		</datafield>
		<datafield tag="700" ind1="1" ind2=" ">
			<subfield code="a">Richards, Angela</subfield>
			<subfield code="9">lat</subfield>
			<subfield code="8">PreferredLanguageHeading</subfield>
			<subfield code="0">https://open-eu.hosted.exlibrisgroup.com/alma/972NNL_INST/authorities/987007272266705171.jsonld</subfield>
			<subfield code="3">NLI-987007272266705171</subfield>
		</datafield>
		<datafield tag="710" ind1="2" ind2=" ">
			<subfield code="a">Institute of Psycho-analysis (Great Britain)</subfield>
			<subfield code="9">lat</subfield>
			<subfield code="8">PreferredLanguageHeading</subfield>
			<subfield code="0">https://open-eu.hosted.exlibrisgroup.com/alma/972NNL_INST/authorities/987007263071705171.jsonld</subfield>
			<subfield code="3">NLI-987007263071705171</subfield>
		</datafield>
		<datafield tag="914" ind1=" " ind2=" ">
			<subfield code="a">AAC</subfield>
		</datafield>
		<datafield tag="925" ind1=" " ind2=" ">
			<subfield code="a">0088216aac</subfield>
		</datafield>
		<datafield tag="999" ind1=" " ind2=" ">
			<subfield code="a">BOOK</subfield>
		</datafield>
		<datafield tag="TAU" ind1="0" ind2="1">
			<subfield code="a">00988216</subfield>
		</datafield>
		<datafield tag="900" ind1=" " ind2=" ">
			<subfield code="a">n 79056228</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007261472805171</subfield>
			<subfield code="4">010</subfield>
		</datafield>
		<datafield tag="900" ind1="7" ind2=" ">
			<subfield code="a">Q78485</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007261472805171</subfield>
			<subfield code="4">024</subfield>
		</datafield>
		<datafield tag="900" ind1="7" ind2=" ">
			<subfield code="a">4930067</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007261472805171</subfield>
			<subfield code="4">024</subfield>
		</datafield>
		<datafield tag="900" ind1=" " ind2=" ">
			<subfield code="a">n 50010691</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007268543505171</subfield>
			<subfield code="4">010</subfield>
		</datafield>
		<datafield tag="900" ind1="7" ind2=" ">
			<subfield code="a">Q1681142</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007268543505171</subfield>
			<subfield code="4">024</subfield>
		</datafield>
		<datafield tag="900" ind1="7" ind2=" ">
			<subfield code="a">36999490</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007268543505171</subfield>
			<subfield code="4">024</subfield>
		</datafield>
		<datafield tag="900" ind1=" " ind2=" ">
			<subfield code="a">n  50053834</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007263071705171</subfield>
			<subfield code="4">010</subfield>
		</datafield>
		<datafield tag="900" ind1="7" ind2=" ">
			<subfield code="a">Q110526506</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007263071705171</subfield>
			<subfield code="4">024</subfield>
		</datafield>
		<datafield tag="900" ind1="7" ind2=" ">
			<subfield code="a">132378264</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007263071705171</subfield>
			<subfield code="4">024</subfield>
		</datafield>
		<datafield tag="900" ind1=" " ind2=" ">
			<subfield code="a">n 50017050</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007463255805171</subfield>
			<subfield code="4">010</subfield>
		</datafield>
		<datafield tag="900" ind1="7" ind2=" ">
			<subfield code="a">Q1328815</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007463255805171</subfield>
			<subfield code="4">024</subfield>
		</datafield>
		<datafield tag="900" ind1="7" ind2=" ">
			<subfield code="a">54189100</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007463255805171</subfield>
			<subfield code="4">024</subfield>
		</datafield>
		<datafield tag="900" ind1=" " ind2=" ">
			<subfield code="a">n 87108304</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007272266705171</subfield>
			<subfield code="4">010</subfield>
		</datafield>
		<datafield tag="900" ind1="7" ind2=" ">
			<subfield code="a">303920731</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007272266705171</subfield>
			<subfield code="4">024</subfield>
		</datafield>
		<datafield tag="900" ind1=" " ind2=" ">
			<subfield code="a">n 50045916</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007259053805171</subfield>
			<subfield code="4">010</subfield>
		</datafield>
		<datafield tag="900" ind1="7" ind2=" ">
			<subfield code="a">Q84430</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007259053805171</subfield>
			<subfield code="4">024</subfield>
		</datafield>
		<datafield tag="900" ind1="7" ind2=" ">
			<subfield code="a">68976881</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007259053805171</subfield>
			<subfield code="4">024</subfield>
		</datafield>
		<datafield tag="900" ind1=" " ind2=" ">
			<subfield code="a">n 85073834</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007280876605171</subfield>
			<subfield code="4">010</subfield>
		</datafield>
		<datafield tag="900" ind1="7" ind2=" ">
			<subfield code="a">Q4727526</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007280876605171</subfield>
			<subfield code="4">024</subfield>
		</datafield>
		<datafield tag="900" ind1="7" ind2=" ">
			<subfield code="a">34536224</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007280876605171</subfield>
			<subfield code="4">024</subfield>
		</datafield>
		<datafield tag="900" ind1=" " ind2=" ">
			<subfield code="a">n 79043849</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007261471505171</subfield>
			<subfield code="4">010</subfield>
		</datafield>
		<datafield tag="900" ind1="7" ind2=" ">
			<subfield code="a">Q9215</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007261471505171</subfield>
			<subfield code="4">024</subfield>
		</datafield>
		<datafield tag="900" ind1="7" ind2=" ">
			<subfield code="a">34456780</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007261471505171</subfield>
			<subfield code="4">024</subfield>
		</datafield>
		<datafield tag="900" ind1=" " ind2=" ">
			<subfield code="a">sh 85108411</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007546289205171</subfield>
			<subfield code="4">010</subfield>
		</datafield>
		<datafield tag="900" ind1="7" ind2=" ">
			<subfield code="a">Q41630</subfield>
			<subfield code="2">NLI</subfield>
			<subfield code="3">NLI-987007546289205171</subfield>
			<subfield code="4">024</subfield>
		</datafield>
	</record>
</collection>
    """
    user_question = "Name the authors XML document, and provide a summary (up to 500 words) of each one of them"
    prompt = f"""
        You are an expert in analyzing data from XML documents. 
        Your task is to extract specific information from the XML content and answer the user's question.

        XML Document Content:
        ```xml
        {xml_string}
        ```

        User's question: {user_question}

        Please provide your answer in a clear and concise manner. If the question is about the number of records,
        please provide the number. If the question is about the names of authors, please provide a list of the authors.
        If the question is about something else, please provide the answer in a clear and concise manner.
    """
    ai_response = get_ai_response(prompt)
    print("AI Response:\n", ai_response, "\n")