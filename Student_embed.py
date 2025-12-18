import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------
# 1. Load embedding model
# ---------------------------
MODEL_NAME = "BAAI/bge-large-en-v1.5"
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# ---------------------------
# 2. Load Student Dataset
# ---------------------------
stu_df = pd.read_excel("Student_Dataset.xlsx")
stu_df.columns = [c.strip().lower().replace(" ", "_") for c in stu_df.columns]

print("Columns detected:", list(stu_df.columns))
print("Total rows:", len(stu_df))

# ---------------------------
# 3. Row-wise semantic sentence creation (YOUR LOGIC)
# ---------------------------
def to_semantic_text(row):
    return (
        f"{row['name']} is a {row['gender']} student from the {row['branch']} branch "
        f"with roll number {row['roll_no']}. "
        f"They are pursuing a {row['degree_name']} degree, "
        f"joined in {row['joining_year']} and passed out in {row['passed_year']}. "
        f"They earned {row['credits']} credits with a CGPA of {row['cgpa']}. "
        f"Admission type was {row['admission']} and got placed in {row['company_placed']}."
    )

stu_df["semantic_text"] = stu_df.apply(to_semantic_text, axis=1)

sentences = stu_df["semantic_text"].tolist()

# ---------------------------
# 4. Generate embeddings
# ---------------------------
print("Generating embeddings...")
embeddings = model.encode(
    sentences,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

# ---------------------------
# 5. Save as JSON (RAG-ready)
# ---------------------------
output = []

for i in range(len(sentences)):
    output.append({
        "text": sentences[i],
        "embedding": embeddings[i].tolist(),
        "metadata": {
            "roll_no": stu_df.iloc[i]["roll_no"],
            "branch": stu_df.iloc[i]["branch"],
            "cgpa": stu_df.iloc[i]["cgpa"]
        }
    })

with open("student_row_wise_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("✅ Embeddings saved to student_row_wise_embeddings.json")
