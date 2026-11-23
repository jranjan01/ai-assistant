import hashlib
import json
import os

import numpy
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

VECTOR_STORE_FILE = os.getenv("VECTOR_STORE_FILE")

KB_FOLDER = os.getenv("KB_FOLDER")

MIN_SCORE = 0.50  # chunks scoring below this are treated as "not relevant"
TOP_K = 2
INDEX_VERSION = 1

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
)


def get_embedding(text: str):
    response = client.embeddings.create(model=os.getenv("EMBEDDING_MODEL"), input=text)
    return response.data[0].embedding


def cosine_similarity(A, B):  # Calculate the cosine similarity between two vectors
    dot_product = numpy.dot(A, B)  # Compute the dot product of vectors A and B
    norm_a = numpy.linalg.norm(A)  # Calculate the length (magnitude) of vector A
    norm_b = numpy.linalg.norm(B)  # Calculate the length (magnitude) of vector B

    return dot_product / (norm_a * norm_b)  # Return the cosine similarity score


# Add embeddings in the chunks
def add_embedding(chunks):
    for chunk in chunks:
        chunk["vector"] = get_embedding(chunk["text"])
    return chunks


def load_documents(folder=KB_FOLDER):
    documents = []
    for filename in os.listdir(folder):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder, filename)

        # open the file
        with open(path, "r", encoding="utf-8") as f:
            documents.append(
                {
                    "source": filename,
                    "text": f.read(),
                }
            )

    return documents


# Takes one file and create multiple chunks from it.
def chunk_document(document):
    chunks = []

    chunk_id = 0
    for part in document["text"].split("\n\n"):
        part = part.strip()
        if part:
            chunks.append(
                {
                    "source": document["source"],
                    "chunk_id": chunk_id,
                    "text": part,
                }
            )
            chunk_id += 1

    return chunks


# loads the knowledge base and create chunks from each document
def create_chunks():
    documents = load_documents()
    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    return chunks


def make_fingerprint(chunks):
    """A hash that changes whenever the documents, model or provider change."""
    data = {
        "version": INDEX_VERSION,
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "chunks": chunks,  # no vectors yet, so only the text and ids are hashed
    }
    text = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cached_index(fingerprint):
    """Return the cached chunks if the cache is valid, otherwise None."""
    try:
        with open(VECTOR_STORE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None  # no cache yet, or the file is broken

    if cache.get("version") != INDEX_VERSION:
        return None  # old cache format
    if cache.get("fingerprint") != fingerprint:
        return None  # documents or settings changed

    return cache.get("chunks")


def save_index(fingerprint, chunks):
    """Save the index safely: write a temp file first, then swap it in."""
    cache = {"version": INDEX_VERSION, "fingerprint": fingerprint, "chunks": chunks}
    temp_file = f"{VECTOR_STORE_FILE}.tmp"

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(
        temp_file, VECTOR_STORE_FILE
    )  # a crash mid-write never leaves a half file


def load_or_create_index():
    """Reuse saved embeddings if nothing changed, otherwise embed again."""
    chunks = create_chunks()
    fingerprint = make_fingerprint(chunks)

    cached_chunks = load_cached_index(fingerprint)
    if cached_chunks:
        return cached_chunks

    print("Building knowledge base index...")
    chunks = add_embedding(chunks)

    try:
        save_index(fingerprint, chunks)
    except OSError as error:
        print(f"Could not save index cache: {error}")  # the app still works without it

    return chunks


# Runs once when the app starts: reuse cached embeddings or build .
KNOWLEDGE_CHUNKS = load_or_create_index()


def search_knowledge_base(query: str):
    """Find the chunks that best match the question."""

    # 1. Turn the question into a vector
    query_vector = get_embedding(query)

    # 2. Score every chunk, keeping only the ones that are close enough
    matches = []
    for chunk in KNOWLEDGE_CHUNKS:
        score = cosine_similarity(query_vector, chunk["vector"])
        if round(score, 2) >= MIN_SCORE:
            matches.append((score, chunk))

    # 3. Nothing relevant: tell the AI there is no information
    if not matches:
        return {
            "found": False,
            "message": "No relevant information found in the knowledge base.",
        }

    # 4. Best match first, keep only the top few
    matches.sort(key=lambda match: match[0], reverse=True)
    top_matches = matches[:TOP_K]

    # 5. Send only source and text to the AI (no score, no vector)
    return {
        "found": True,
        "results": [
            {"source": chunk["source"], "text": chunk["text"]}
            for _, chunk in top_matches
        ],
    }
