import os
import requests
from urllib3.exceptions import InsecureRequestWarning
import urllib3

# Disable SSL warnings (optional)
urllib3.disable_warnings(InsecureRequestWarning)

# Directory to save PDFs
DOWNLOAD_DIR = "financial_pdfs"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Fake browser header
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

# List of (filename, URL) tuples
pdf_files = [
    ("JPM_Annual_2023.pdf", "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/annualreport-2023.pdf"),
    ("JPM_Annual_2024.pdf", "https://www.sec.gov/Archives/edgar/data/19617/000001961725000329/annualreport-2024.pdf"),
    ("DB_Annual_2023.pdf", "https://investor-relations.db.com/files/documents/annual-reports/2023/20-F-2023.pdf"),
    ("JPM_SE_Annual_2023.pdf", "https://www.jpmorgan.com/content/dam/jpm/global/disclosures/de/english-version-of-disclosures/2023-annual-report-english.pdf"),
    ("Unilever_Annual_2024.pdf", "https://www.unilever.com/files/unilever-annual-report-on-form-20-f-2024.pdf"),
]

def download_pdf(name, url):
    path = os.path.join(DOWNLOAD_DIR, name)
    try:
        print(f"Downloading {name}...")
        response = requests.get(url, headers=headers, stream=True, timeout=20, verify=False)
        response.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Saved to {path}")
    except Exception as e:
        print(f"❌ Failed to download {name}: {e}")

# Download all files
for name, url in pdf_files:
    download_pdf(name, url)