import re
import nltk
from nltk.stem import WordNetLemmatizer
import stanza
from typing import Iterator, List

# Download required data once (should be done outside of library code in production)
stanza.download('en')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# Initialize NLP tools
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma')
lemmatizer = WordNetLemmatizer()

def preprocess_line(line: str, special_symbol=None) -> str:
    """Lowercase, strip, and optionally remove special symbols from a line."""
    line = line.strip().lower()
    if special_symbol:
        line = special_symbol.sub('', line)
    line = re.sub(r'\s+', ' ', line)  # normalize spacing
    return line


def pre_process_text_stream(page: Iterator[str]) -> Iterator[str]:
    """Yield cleaned sentences from a stream of lines."""
    previous_line = ''
    sentence_end_regex = re.compile(r'([^.!?]*[.!?])')
    square_six = re.compile(r'/square6')
    for line in page:
        line = preprocess_line(line, square_six)
        if not line:
            continue
        line = previous_line + " " + line if previous_line else line
        matches = sentence_end_regex.findall(line)
        if matches:
            for i, sentence in enumerate(matches):
                sentence = sentence.strip()
                if i == len(matches) - 1 and not sentence.endswith(('.', '!', '?')):
                    previous_line = sentence
                else:
                    previous_line = ""
                    yield sentence
        else:
            previous_line += " " + line
    if previous_line.strip():
        yield previous_line.strip()


def tokenize_stream(stream: Iterator[str]) -> Iterator[List[str]]:
    """Tokenize and lemmatize each line in a stream."""
    word = re.compile(r'\W')
    number_substitution = re.compile(r'[0-9]+\.[0-9]+')
    for line in stream:
        line = number_substitution.sub('', line)
        line = word.sub(' ', line)
        line = line.split(' ')
        lemmas = [lemmatizer.lemmatize(token) for token in line if token]
        yield lemmas


def tokenize(query: str) -> List[str]:
    """Tokenize and lemmatize a single query string using stanza."""
    line = preprocess_line(query)
    word = re.compile(r'\W')
    number_substitution = re.compile(r'[0-9]+\.[0-9]+')
    line = number_substitution.sub('', line)
    line = word.sub(' ', line)
    doc = nlp(line)
    lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
    return lemmas