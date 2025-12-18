import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "BAAI/bge-large-en-v1.5"

CHUNKS_FILE = "chunks.json"
MEDIA_FILE = "media.json"

TEXT_OUTPUT = "chunks_embeddings.json"
MEDIA_OUTPUT = "media_embeddings.json"

# -----------------------------
# LOAD MODEL
# -----------------------------
print("🔹 Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# -----------------------------
# HELPERS
# -----------------------------
def valid_text(text: str) -> bool:
    return isinstance(text, str) and len(text.strip()) > 40


def embed(text: str):
    return model.encode(text, normalize_embeddings=True).tolist()


def generate_image_text(source_page: str) -> str:
    """
    SAFE semantic text generation from metadata
    """
    return (
        f"Image associated with informational content from the webpage {source_page}."
    )

# =============================
# COLLECTION 1: TEXT CHUNKS
# =============================
print("🔹 Loading chunks.json...")
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

text_collection = []

print(f"🔹 Embedding {len(chunks)} text chunks...")
for item in tqdm(chunks, desc="Text embeddings"):
    text = item.get("text", "")
    metadata = item.get("metadata", {})

    if not valid_text(text):
        continue

    vector = embed(text)

    text_collection.append({
        "id": f"chunk_{metadata.get('chunk_id')}",
        "page_url": metadata.get("page_url"),
        "embedding": vector
    })

# SAVE TEXT COLLECTION
with open(TEXT_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(text_collection, f, indent=2)

print(f"✅ Text collection saved: {TEXT_OUTPUT}")
print(f"✅ Total text embeddings: {len(text_collection)}")

# =============================
# COLLECTION 2: IMAGE METADATA
# =============================
print("🔹 Loading media.json...")
with open(MEDIA_FILE, "r", encoding="utf-8") as f:
    media = json.load(f)

media_collection = []

print(f"🔹 Embedding {len(media)} images (via generated text)...")
for idx, item in enumerate(tqdm(media, desc="Image embeddings")):
    source_page = item.get("source_page")
    if not source_page:
        continue

    image_text = generate_image_text(source_page)
    vector = embed(image_text)

    media_collection.append({
        "id": f"image_{idx}",
        "page_url": source_page,
        "embedding": vector
    })

# SAVE MEDIA COLLECTION
with open(MEDIA_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(media_collection, f, indent=2)

print(f"✅ Media collection saved: {MEDIA_OUTPUT}")
print(f"✅ Total image embeddings: {len(media_collection)}")

# =============================
# FINAL CHECK
# =============================
if text_collection:
    print("🔍 Text embedding dimension:", len(text_collection[0]["embedding"]))
if media_collection:
    print("🔍 Image embedding dimension:", len(media_collection[0]["embedding"]))

print("🎯 DONE — Two clean collections ready for RAG")
