import os
from PyPDF2 import PdfReader

# Directory containing all PDF files
pdf_dir = os.path.join(os.path.dirname(__file__), 'paper')

# List all PDF files in the directory
pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_dir, pdf_file)
    txt_path = os.path.splitext(pdf_path)[0] + '.txt'
    try:
        reader = PdfReader(pdf_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Extracted text to: {txt_path}")
    except Exception as e:
        print(f"Failed to extract {pdf_file}: {e}")
