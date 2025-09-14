# requirement_analyzer.py
import spacy

# Load small English model
nlp = spacy.load("en_core_web_sm")

def analyze_requirements(text):
    """
    Extracts key features from requirement text using NLP.
    Example: turns 'The system should allow user login with email'
    into ['system', 'user login', 'email'].
    """
    doc = nlp(text)
    keywords = [chunk.text for chunk in doc.noun_chunks]
    return list(set(keywords))  # unique keywords


if __name__ == "__main__":
    sample_text = "The system should allow users to login using email and password."
    print("Input:", sample_text)
    print("Extracted Features:", analyze_requirements(sample_text))
