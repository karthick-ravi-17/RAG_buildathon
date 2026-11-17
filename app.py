#python -m pip install streamlit langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu pypdf python-dotenv
#python -m pip install playwright beautifulsoup4
#python -m playwright install chromium
#streamlit run rag_app.py
#https://aed.tn.gov.in/en/individual-based-subsidy-schemes/

import os
import re
import json
import shutil
import tempfile
import ipaddress
import socket

import streamlit as st

from urllib.parse import urlparse
from dotenv import load_dotenv

from playwright.sync_api import sync_playwright

from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# ==================================================
# 1. CONFIGURATION
# ==================================================

load_dotenv()

st.set_page_config(
    page_title="Multi-Source RAG Assistant",
    page_icon="📚",
    layout="wide"
)

API_KEY = os.getenv("OPENAI_API_KEY")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 5

FAISS_INDEX_DIR = "faiss_index"
PDF_STORAGE_DIR = "uploaded_pdfs"

os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)


# ==================================================
# 2. SESSION STATE INITIALIZATION
# ==================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "indexed_urls" not in st.session_state:
    st.session_state.indexed_urls = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False


# ==================================================
# 3. OPENAI EMBEDDINGS
# ==================================================

def get_embeddings():

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=API_KEY
    )


# ==================================================
# 4. URL VALIDATION
# ==================================================

def validate_website_url(url):

    parsed_url = urlparse(url)

    if parsed_url.scheme not in ["http", "https"]:
        raise ValueError(
            f"Only HTTP and HTTPS URLs are supported: {url}"
        )

    if not parsed_url.hostname:
        raise ValueError(
            f"Invalid website URL: {url}"
        )

    hostname = parsed_url.hostname.lower()

    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError(
            "Localhost URLs are not allowed."
        )

    try:

        ip_addresses = socket.getaddrinfo(
            hostname,
            None
        )

        for address in ip_addresses:

            ip = ipaddress.ip_address(
                address[4][0]
            )

            if not ip.is_global:

                raise ValueError(
                    f"Private or restricted IP addresses are not allowed: {url}"
                )

    except socket.gaierror:

        raise ValueError(
            f"Unable to resolve website: {url}"
        )

    return True


# ==================================================
# 5. LOAD PERSISTED FAISS DATABASE
# ==================================================

def load_persisted_index():

    index_path = os.path.join(
        FAISS_INDEX_DIR,
        "index.faiss"
    )

    pickle_path = os.path.join(
        FAISS_INDEX_DIR,
        "index.pkl"
    )

    manifest_path = os.path.join(
        FAISS_INDEX_DIR,
        "manifest.json"
    )

    if not (
        os.path.exists(index_path)
        and os.path.exists(pickle_path)
    ):
        return False

    try:

        embeddings = get_embeddings()

        vector_store = FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

        st.session_state.vector_store = vector_store

        if os.path.exists(manifest_path):

            with open(manifest_path, "r") as file:

                manifest = json.load(file)

            st.session_state.chunk_count = manifest.get(
                "chunk_count", 0
            )

            st.session_state.indexed_files = manifest.get(
                "indexed_files", []
            )

            st.session_state.indexed_urls = manifest.get(
                "indexed_urls", []
            )

        st.session_state.index_loaded = True

        return True

    except Exception as error:

        st.warning(
            f"Unable to restore FAISS database: {error}"
        )

        return False


# ==================================================
# 6. AUTOMATICALLY RESTORE DATABASE
# ==================================================

if (
    st.session_state.vector_store is None
    and API_KEY
    and not st.session_state.index_loaded
):

    load_persisted_index()


# ==================================================
# 7. LOAD PDF DOCUMENTS
# ==================================================

def load_pdf_documents(uploaded_files):

    all_documents = []

    for uploaded_file in uploaded_files:

        safe_filename = os.path.basename(
            uploaded_file.name
        )

        pdf_path = os.path.join(
            PDF_STORAGE_DIR,
            safe_filename
        )

        with open(pdf_path, "wb") as file:

            file.write(
                uploaded_file.getvalue()
            )

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        for document in documents:

            document.metadata["source_type"] = "PDF"

            document.metadata["file_name"] = (
                uploaded_file.name
            )

            document.metadata["page_number"] = (
                document.metadata.get("page", 0) + 1
            )

            document.metadata["section"] = (
                document.metadata.get(
                    "section",
                    "Not detected"
                )
            )

            document.metadata["source"] = (
                uploaded_file.name
            )

        all_documents.extend(documents)

    return all_documents


# ==================================================
# 8. SCRAPE WEBSITE USING PLAYWRIGHT
# ==================================================

def scrape_website(url):

    validate_website_url(url)

    documents = []

    with sync_playwright() as playwright:

        browser = playwright.chromium.launch(
            headless=True
        )

        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )

        try:

            response = page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30000
            )

            if response is None or not response.ok:

                status = (
                    response.status if response else "No response"
                )

                raise ValueError(
                    f"Website returned an unsuccessful response: {status}"
                )

            # Allow JavaScript-rendered content to load
            page.wait_for_timeout(1500)

            # Extract page title
            page_title = page.title()

            # Extract visible text from the page
            page_content = page.locator(
                "body"
            ).inner_text(timeout=15000)

            # Basic cleanup
            page_content = re.sub(
                r"\n{3,}",
                "\n\n",
                page_content
            )

            page_content = page_content.strip()

            if not page_content:

                raise ValueError(
                    "No readable content found on the website."
                )

            # Create LangChain document
            website_document = Document(

                page_content=page_content,

                metadata={

                    "source_type": "Website",

                    "url": url,

                    "source": url,

                    "file_name": page_title or url,

                    "page_number": "N/A",

                    "section": page_title or "Website content"

                }

            )

            documents.append(
                website_document
            )

        finally:

            browser.close()

    return documents


# ==================================================
# 9. LOAD ALL DATA SOURCES
# ==================================================

def load_all_documents(
    uploaded_files,
    website_urls
):

    all_documents = []

    # Process PDF documents
    if uploaded_files:

        pdf_documents = load_pdf_documents(
            uploaded_files
        )

        all_documents.extend(
            pdf_documents
        )

    # Process website URLs
    for url in website_urls:

        website_documents = scrape_website(
            url
        )

        all_documents.extend(
            website_documents
        )

    return all_documents


# ==================================================
# 10. SPLIT DOCUMENTS INTO CHUNKS
# ==================================================

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP

    )

    chunks = text_splitter.split_documents(
        documents
    )

    return chunks


# ==================================================
# 11. CREATE FAISS VECTOR DATABASE
# ==================================================

def create_vector_store(chunks):

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(

        documents=chunks,

        embedding=embeddings

    )

    return vector_store


# ==================================================
# 12. BUILD AND PERSIST RAG INDEX
# ==================================================

def build_rag_index(
    uploaded_files,
    website_urls
):

    if not uploaded_files and not website_urls:

        raise ValueError(
            "Upload PDFs or provide at least one website URL."
        )

    with st.spinner(
        "Extracting content from all sources..."
    ):

        # Clean previous PDF storage
        if os.path.exists(PDF_STORAGE_DIR):

            shutil.rmtree(PDF_STORAGE_DIR)

        os.makedirs(
            PDF_STORAGE_DIR,
            exist_ok=True
        )

        # Extract content from PDFs and websites
        documents = load_all_documents(
            uploaded_files,
            website_urls
        )

        if not documents:

            raise ValueError(
                "No readable content found."
            )

        # Split all documents together
        chunks = split_documents(
            documents
        )

        if not chunks:

            raise ValueError(
                "No chunks generated."
            )

        # Create one combined FAISS database
        vector_store = create_vector_store(
            chunks
        )

        # Persist FAISS database
        vector_store.save_local(
            FAISS_INDEX_DIR
        )

        # Save index information
        manifest = {

            "chunk_count": len(chunks),

            "indexed_files": [
                file.name for file in uploaded_files
            ],

            "indexed_urls": website_urls

        }

        manifest_path = os.path.join(
            FAISS_INDEX_DIR,
            "manifest.json"
        )

        with open(manifest_path, "w") as file:

            json.dump(
                manifest,
                file,
                indent=4
            )

        # Update Streamlit session state
        st.session_state.vector_store = vector_store

        st.session_state.chunk_count = len(chunks)

        st.session_state.indexed_files = (
            manifest["indexed_files"]
        )

        st.session_state.indexed_urls = (
            manifest["indexed_urls"]
        )

        st.session_state.index_loaded = True

        st.session_state.chat_history = []

        return len(chunks)


# ==================================================
# 13. RETRIEVE RELEVANT DOCUMENTS
# ==================================================

def retrieve_documents(question):

    vector_store = st.session_state.vector_store

    if vector_store is None:

        raise ValueError(
            "FAISS database is not available."
        )

    relevant_documents = vector_store.similarity_search(

        question,

        k=RETRIEVAL_K

    )

    return relevant_documents


# ==================================================
# 14. GENERATE ANSWER USING OPENAI
# ==================================================

def generate_answer(question):

    relevant_documents = retrieve_documents(
        question
    )

    context_parts = []

    for index, document in enumerate(
        relevant_documents,
        start=1
    ):

        metadata = document.metadata

        if metadata.get("source_type") == "PDF":

            source_information = f"""
            Source Type: PDF
            File Name: {metadata.get("file_name")}
            Page Number: {metadata.get("page_number")}
            Section: {metadata.get("section")}
            """

        else:

            source_information = f"""
            Source Type: Website
            Website URL: {metadata.get("url")}
            Page Title: {metadata.get("file_name")}
            """

        context_parts.append(

            f"""
            [SOURCE {index}]

            {source_information}

            Content:

            {document.page_content}
            """

        )

    context = "\n\n".join(
        context_parts
    )

    llm = ChatOpenAI(

        model="gpt-4o-mini",

        temperature=0,

        api_key=API_KEY

    )

    prompt = f"""
    You are a multi-source RAG question-answering assistant.

    Answer the user's question using ONLY the
    retrieved document context.

    Instructions:

    1. Do not invent facts or information.

    2. If the answer is not available in the
       retrieved context, clearly say so.

    3. Cite every factual claim using
       [SOURCE 1], [SOURCE 2], etc.

    4. For PDF sources, identify the filename
       and page number.

    5. For website sources, identify the
       website URL and page title.

    6. Never fabricate source information.

    7. If multiple sources support the answer,
       cite all relevant sources.

    Retrieved Context:

    {context}

    User Question:

    {question}

    Answer:
    """

    response = llm.invoke(
        prompt
    )

    return response.content, relevant_documents


# ==================================================
# 15. STREAMLIT USER INTERFACE
# ==================================================

st.title("📚 Multi-Source RAG Assistant")

st.write(
    """
    Upload PDF documents, provide website URLs,
    or combine both sources to ask questions
    using AI-powered document retrieval.
    """
)


# ==================================================
# 16. SIDEBAR - DATA SOURCE INPUT
# ==================================================

with st.sidebar:

    st.header("📂 Data Sources")

    st.write("**1. Upload PDF documents**")

    uploaded_files = st.file_uploader(

        "Choose PDF files",

        type=["pdf"],

        accept_multiple_files=True

    )

    st.divider()

    st.write("**2. Provide Website URLs**")

    website_input = st.text_area(

        "Enter website URLs (one URL per line)",

        placeholder=(
            "https://example.com\n"
            "https://example.com/about"
        ),

        height=100

    )

    website_urls = [

        url.strip()

        for url in website_input.splitlines()

        if url.strip()

    ]

    st.caption(
        "You can provide PDFs, URLs, or both."
    )

    st.divider()

    if st.button(

        "Build RAG Database",

        use_container_width=True

    ):

        if not API_KEY:

            st.error(
                "OPENAI_API_KEY not found in .env file."
            )

        elif not uploaded_files and not website_urls:

            st.warning(
                "Please upload a PDF or provide a website URL."
            )

        else:

            try:

                chunk_count = build_rag_index(

                    uploaded_files,

                    website_urls

                )

                st.success(
                    f"Successfully indexed {chunk_count} chunks!"
                )

                st.info(
                    f"PDF documents: {len(uploaded_files)}"
                )

                st.info(
                    f"Website URLs: {len(website_urls)}"
                )

            except Exception as error:

                st.error(
                    f"Indexing failed: {str(error)}"
                )


    # ==================================================
    # 17. DISPLAY INDEX INFORMATION
    # ==================================================

    if st.session_state.vector_store is not None:

        st.divider()

        st.subheader("📊 FAISS Information")

        st.success(
            "Vector database is ready!"
        )

        st.metric(

            "Total Chunks Indexed",

            st.session_state.chunk_count

        )

        st.write("**Indexed PDFs:**")

        if st.session_state.indexed_files:

            for file_name in st.session_state.indexed_files:

                st.write(f"📄 {file_name}")

        else:

            st.caption("No PDF documents indexed.")

        st.write("**Indexed Websites:**")

        if st.session_state.indexed_urls:

            for url in st.session_state.indexed_urls:

                st.write(f"🌐 {url}")

        else:

            st.caption("No websites indexed.")


# ==================================================
# 18. CHAT INTERFACE
# ==================================================

st.subheader("💬 Ask Questions")

if st.session_state.vector_store is None:

    st.info(
        "Build your RAG database before asking questions."
    )

else:

    # Display chat history
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(
                message["content"]
            )

            if message["role"] == "assistant":

                for source in message.get(
                    "sources",
                    []
                ):

                    if source["type"] == "PDF":

                        source_label = (
                            f"📄 {source['file']} | "
                            f"Page {source['page']}"
                        )

                    else:

                        source_label = (
                            f"🌐 {source['title']}"
                        )

                    with st.expander(source_label):

                        if source["type"] == "PDF":

                            st.write(
                                f"File: {source['file']}"
                            )

                            st.write(
                                f"Page: {source['page']}"
                            )

                            st.write(
                                f"Section: {source['section']}"
                            )

                        else:

                            st.write(
                                f"Website: {source['url']}"
                            )

                            st.write(
                                f"Page Title: {source['title']}"
                            )

                            st.markdown(
                                f"[Open Website]({source['url']})"
                            )

                        st.write(
                            source["content"]
                        )

    user_question = st.chat_input(

        "Ask a question about your PDFs or websites..."

    )

    if user_question:

        st.session_state.chat_history.append({

            "role": "user",

            "content": user_question

        })

        with st.chat_message("user"):

            st.markdown(
                user_question
            )

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching sources and generating answer..."
            ):

                try:

                    answer, source_documents = generate_answer(
                        user_question
                    )

                    st.markdown(answer)

                    sources = []

                    for document in source_documents:

                        metadata = document.metadata

                        if metadata.get("source_type") == "PDF":

                            sources.append({

                                "type": "PDF",

                                "file": metadata.get(
                                    "file_name",
                                    "Unknown"
                                ),

                                "page": metadata.get(
                                    "page_number",
                                    "Unknown"
                                ),

                                "section": metadata.get(
                                    "section",
                                    "Not detected"
                                ),

                                "content": document.page_content

                            })

                        else:

                            sources.append({

                                "type": "Website",

                                "url": metadata.get(
                                    "url",
                                    "Unknown"
                                ),

                                "title": metadata.get(
                                    "file_name",
                                    "Website"
                                ),

                                "content": document.page_content

                            })

                    # Display sources
                    st.markdown(
                        "### 📚 Answer Sources"
                    )

                    for index, source in enumerate(
                        sources,
                        start=1
                    ):

                        if source["type"] == "PDF":

                            label = (
                                f"Source {index}: "
                                f"{source['file']} | "
                                f"Page {source['page']}"
                            )

                        else:

                            label = (
                                f"Source {index}: "
                                f"{source['title']}"
                            )

                        with st.expander(label):

                            if source["type"] == "PDF":

                                st.write(
                                    f"File: {source['file']}"
                                )

                                st.write(
                                    f"Page: {source['page']}"
                                )

                                st.write(
                                    f"Section: {source['section']}"
                                )

                            else:

                                st.write(
                                    f"Website: {source['url']}"
                                )

                                st.markdown(
                                    f"[Open Website]({source['url']})"
                                )

                            st.write(
                                source["content"]
                            )

                    # Save assistant response
                    st.session_state.chat_history.append({

                        "role": "assistant",

                        "content": answer,

                        "sources": sources

                    })

                except Exception as error:

                    error_message = (
                        f"Error generating answer: {str(error)}"
                    )

                    st.error(
                        error_message
                    )

                    st.session_state.chat_history.append({

                        "role": "assistant",

                        "content": error_message,

                        "sources": []

                    })
