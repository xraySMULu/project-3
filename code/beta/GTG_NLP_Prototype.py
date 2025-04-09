import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import textwrap

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from each page of a PDF file using PyPDF2.
    Returns a single string containing the PDF's text.
    """
    text_content = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
    return "\n".join(text_content)

def clean_and_tokenize(text):
    """
    1. Lowercase the text
    2. Remove non-alphabetic characters
    3. Split into tokens
    4. Remove stopwords
    5. Lemmatize each token
    """
    # Lowercase
    text = text.lower()
    
    # Remove non-alphabetic characters (numbers, punctuation, etc.)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Split into tokens
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def prepare_tfidf_features(token_lists):
    """
    Convert lists of tokens (documents) into TF-IDF vectors.
    For demonstration, we simply join tokens back into strings
    before feeding them to TfidfVectorizer.
    """
    # Join each list of tokens back into a single string
    documents = [" ".join(tokens) for tokens in token_lists]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    return tfidf_matrix, vectorizer
def generate_pdf_of_processed_data(token_lists, output_filename):
    """
    Creates a PDF containing the processed tokens for the first (or multiple) documents.
    Adjust as needed to display additional results such as TF-IDF terms, etc.
    """
    # First doc's tokens
    if not token_lists:
        print("No token lists found; skipping PDF generation.")
        return
    
    tokens_to_print = token_lists[0]

    # Join tokens back into a single string, and wrap lines so they fit on the page
    text_body = " ".join(tokens_to_print)
    wrapped_text = textwrap.wrap(text_body, width=80)
    
    # Create a new PDF
    c = canvas.Canvas(output_filename, pagesize=letter)
    c.drawString(72, 750, "Processed Data (Tokens) - Document 1")

    # Write text line by line
    y_position = 730
    for line in wrapped_text:
        c.drawString(72, y_position, line)
        y_position -= 15
        # If we reach the bottom of the page, create a new page
        if y_position < 72:
            c.showPage()
            c.drawString(72, 750, "Processed Data (Tokens) - Continued")
            y_position = 730

    c.save()
    print(f"PDF report generated: {output_filename}")


def main(pdf_path, output_pdf_path):
    # 1. Extract text from the PDF
    raw_text = extract_text_from_pdf(pdf_path)
    
    documents = [raw_text]
    
    # 3. Clean and tokenize
    token_lists = [clean_and_tokenize(doc) for doc in documents]
    
    # 4. Convert to TF-IDF features
    tfidf_matrix, vectorizer = prepare_tfidf_features(token_lists)
    
    # 5. Generate a PDF
    generate_pdf_of_processed_data(token_lists, output_pdf_path)
    
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    print("Sample features:", vectorizer.get_feature_names_out()[:20])  # First 20 features
    

if __name__ == "__main__":
    pdf_file_path = "TAOTSChapter1.pdf"
    output_pdf_file = "TAOTSChapter1Processed.pdf"
    main(pdf_file_path, output_pdf_file)
