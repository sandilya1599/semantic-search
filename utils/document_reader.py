from pypdf import PdfReader
import os

def read_documents_stream(folder):
    for dirpath, _, filenames in os.walk(folder):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                if filename.lower().endswith('pdf'):
                    yield from read_document(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

def read_document(document):
    reader = PdfReader(document)
    print('Reading Document: ', document)
    number_of_pages = len(reader.pages)

    for page_number in range(number_of_pages):
        page = reader.pages[page_number]
        text = page.extract_text()
        print('yielding page: ', page_number)
        yield text[:-2]
    print(f"Reading Document: {document} Complete")