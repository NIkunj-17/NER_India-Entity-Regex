from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import spacy
import logging

# Load spaCy's pre-trained model
nlp = spacy.load("en_core_web_sm")

# Indian-specific entity patterns (compiled for better performance)
AADHAAR_PATTERN = re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b')
PAN_PATTERN = re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b')
ATM_CARD_PATTERN = re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b')
DRIVING_LICENSE_PATTERN = re.compile(r'\b[A-Z]{2}\d{2}\s?\d{4}\s?\d{4}\b')
PASSPORT_PATTERN = re.compile(r'\b[A-Z]{1}\d{7}\b')
VOTER_ID_PATTERN = re.compile(r'\b[A-Z]{3}\d{7}\b')

# Entity levels for redaction
ENTITY_LEVELS = {
    1: ['PERSON', 'DATE', 'GPE', 'EMAIL', 'PHONE', 'LOC', 'ADDRESS', 'AGE', 'GENDER', 'TITLE'],
    2: ['CARDINAL', 'MONEY', 'PERCENT', 'FAC', 'ORG', 'PRODUCT', 'TIME', 'QUANTITY'],
    3: ['IDENTIFIER', 'AADHAAR_NUMBER', 'PAN_NUMBER', 'ATM_CARD_NUMBER', 'DRIVING_LICENSE_NUMBER', 'PASSPORT_NUMBER', 'VOTER_ID', 'TITLE', 'OCCUPATION'],
}

# Define a dictionary to map entity labels to redaction strings
ENTITY_MAP = {
    'PERSON': '[REDACTED NAME]',
    'ORG': '[REDACTED ORG]',
    'GPE': '[REDACTED LOCATION]',
    'DATE': '[REDACTED DATE]',
    'EMAIL': '[REDACTED EMAIL]',
    'PHONE': '[REDACTED PHONE]',
    'LOC': '[REDACTED LOC]',
    'ADDRESS': '[REDACTED ADDRESS]',
    'AGE': '[REDACTED AGE]',
    'GENDER': '[REDACTED GENDER]',
    'TITLE': '[REDACTED TITLE]',
    'IDENTIFIER': '[REDACTED IDENTIFIER]',
    'AADHAAR_NUMBER': 'xxx-xxx-xxx',
    'PAN_NUMBER': lambda x: f"XXXXX{x[5:9]}X",
    'ATM_CARD_NUMBER': 'xxxx-xxxx-xxxx-xxxx',
    'DRIVING_LICENSE_NUMBER': 'DL-XXXX XXXX XXXX',
    'PASSPORT_NUMBER': 'PXXXXXXX',
    'VOTER_ID': 'VID-XXXXXXX',
}

app = FastAPI()

class RedactionRequest(BaseModel):
    text: str
    level: int
    custom_placeholders: dict = None  # Allow custom placeholders

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_all_matches(text, doc, level, custom_placeholders):
    """
    Combines spaCy entities and regex matches into a single list of redactions.
    """
    matches = []
    
    # Step 1: Collect spaCy entity matches based on redaction level
    for lvl in range(1, level + 1):
        for ent in doc.ents:
            if ent.label_ in ENTITY_LEVELS[lvl]:
                matches.append((ent.start_char, ent.end_char, ENTITY_MAP.get(ent.label_, '[REDACTED]')))
    
    # Step 2: Collect regex pattern matches (for level 3 redactions)
    if level == 3:
        regex_patterns = [
            (AADHAAR_PATTERN, custom_placeholders.get('AADHAAR_NUMBER', ENTITY_MAP['AADHAAR_NUMBER'])),
            (PAN_PATTERN, custom_placeholders.get('PAN_NUMBER', ENTITY_MAP['PAN_NUMBER'])),
            (ATM_CARD_PATTERN, custom_placeholders.get('ATM_CARD_NUMBER', ENTITY_MAP['ATM_CARD_NUMBER'])),
            (DRIVING_LICENSE_PATTERN, custom_placeholders.get('DRIVING_LICENSE_NUMBER', ENTITY_MAP['DRIVING_LICENSE_NUMBER'])),
            (PASSPORT_PATTERN, custom_placeholders.get('PASSPORT_NUMBER', ENTITY_MAP['PASSPORT_NUMBER'])),
            (VOTER_ID_PATTERN, custom_placeholders.get('VOTER_ID', ENTITY_MAP['VOTER_ID']))
        ]
        
        # Apply regex and collect matches
        for pattern, replacement in regex_patterns:
            for match in re.finditer(pattern, text):
                matches.append((match.start(), match.end(), replacement))
    
    # Step 3: Sort matches by their positions (start_char) to avoid overlaps
    matches.sort(key=lambda x: x[0])
    
    return matches

def apply_redactions(text, matches):
    """
    Applies the redactions to the text in one go.
    """
    redacted_text = []
    last_idx = 0
    
    # Iterate through matches and build the redacted text
    for start, end, replacement in matches:
        redacted_text.append(text[last_idx:start])  # Append text before the match
        if callable(replacement):  # Handle lambda-based replacements
            redacted_text.append(replacement(text[start:end]))
        else:
            redacted_text.append(replacement)
        last_idx = end  # Move to the end of the current match
    
    # Append any remaining text after the last match
    redacted_text.append(text[last_idx:])
    
    return ''.join(redacted_text)

def redact_indian_entities(text: str, level: int, custom_placeholders: dict = None) -> str:
    """
    Redact spaCy entities and Indian-specific identifiers in one pass.
    """
    doc = nlp(text)
    
    # Step 1: Get all matches (spaCy entities + regex patterns)
    matches = get_all_matches(text, doc, level, custom_placeholders)
    
    # Step 2: Apply redactions in one pass
    return apply_redactions(text, matches)

@app.post("/redact")
def redact(request: RedactionRequest):
    try:
        logging.info(f"Received redaction request with level {request.level}")
        redacted_text = redact_indian_entities(request.text, request.level, custom_placeholders=request.custom_placeholders)
        return {"redacted_text": redacted_text}
    except HTTPException as e:
        logging.error(f"Error occurred: {e.detail}")
        raise e
