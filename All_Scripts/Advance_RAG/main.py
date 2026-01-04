import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from embedding_setup import get_azure_embedding_model, get_gemini_embedding_model
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# === Load Azure credentials ===
load_dotenv(override=True)

def assign_roles_to_files(files):
    """Assign roles based on order: 1st analyst, 2nd scientist, 3rd financial."""
    role_map = {}
    for i, file in enumerate(files):
        if i == 0:
            role_map[file] = "analyst"
        elif i == 1:
            role_map[file] = "scientist"
        else:
            role_map[file] = "financial"
    return role_map


def load_data_from_folder(folder_path):
    """Load all PDFs from a folder and attach file_name + role metadata with logs."""
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pdf")])
    role_mapping = assign_roles_to_files(files)

    documents = []
    for file in files:
        pdf_path = os.path.join(folder_path, file)
        assigned_role = role_mapping[file]
        print(f"📄 Loading & embedding file: {file} | Assigned Role: {assigned_role}")

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        # attach metadata
        for d in docs:
            d.metadata["file_name"] = file
            d.metadata["role"] = assigned_role
        documents.extend(docs)

        print(f"✅ Loaded {len(docs)} pages from {file}")
    return documents


# === Chunk documents ===
def chunk_documents(documents, chunk_size_words=2500, chunk_overlap_words=250):
    chunk_size_chars = chunk_size_words * 6
    chunk_overlap_chars = chunk_overlap_words * 6

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_overlap_chars,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)



def create_and_save_faiss_batched(documents, embedding_model, save_path, batch_size=5, sleep_time=5):
    """Embed docs in batches to avoid Azure rate-limit."""
   
    all_embeddings = []
    all_docs = []

    print(f"⚡ Embedding {len(documents)} chunks in batches of {batch_size}...")
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            db = FAISS.from_documents(batch, embedding_model)
            if not all_docs:
                # initialize
                vectorstore = db
            else:
                vectorstore.merge_from(db)
            all_docs.extend(batch)
            print(f"   ✅ Embedded batch {i//batch_size + 1} ({len(batch)} docs)")
        except Exception as e:
            print(f"   ⚠️ Error in batch {i//batch_size + 1}: {e}. Retrying after {sleep_time} sec...")
            time.sleep(sleep_time)
            continue
        time.sleep(sleep_time)  # wait to respect rate limits

    vectorstore.save_local(save_path)
    print(f"✅ FAISS index saved at: {save_path}")


# === Main execution ===
if __name__ == '__main__':
    folder_path = r"/Users/sachinmishra/Desktop/LangGraph_Project_Based_Learning/financial_pdfs"
    faiss_save_path = "faiss_index_financial"

    documents = load_data_from_folder(folder_path)
    print(f"\n📊 Total: {len(documents)} pages loaded from {len(set(d.metadata['file_name'] for d in documents))} PDFs\n")

    chunks = chunk_documents(documents, chunk_size_words=2500)
    print(f"✂️ Created {len(chunks)} chunks with role & file_name metadata")

    embedding_model = get_gemini_embedding_model()
    create_and_save_faiss_batched(chunks, embedding_model, faiss_save_path, batch_size=5, sleep_time=10)

