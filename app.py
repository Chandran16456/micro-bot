import os
import numpy as np
import faiss
from flask import Flask, render_template, request, jsonify
import ollama
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

document_text = ""
document_chunks = []
document_embeddings = []
faiss_index = None
uploaded_filename = None

def split_text_into_chunks(text, chunk_size=800, overlap=150):
    chunks = []

    if not text.strip():
        return chunks

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def create_faiss_index(embeddings):
    embeddings_array = np.array(embeddings).astype("float32")
    dimension = embeddings_array.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    return index

def retrieve_relevant_chunks(query, top_k=3):
    global document_chunks, faiss_index

    if not document_chunks or faiss_index is None:
        return [], []

    query_embedding = embedding_model.encode([query])
    query_array = np.array(query_embedding).astype("float32")

    distances, indices = faiss_index.search(query_array, top_k)

    relevant_chunks = []
    used_indices = []

    for idx in indices[0]:
        if 0 <= idx < len(document_chunks):
            relevant_chunks.append(document_chunks[idx])
            used_indices.append(int(idx))

    return relevant_chunks, used_indices

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global document_text, document_chunks, document_embeddings, faiss_index, uploaded_filename

    data = request.get_json()
    user_message = data.get("message", "")

    try:
        relevant_chunks, used_indices = retrieve_relevant_chunks(user_message, top_k=3)

        if relevant_chunks:
            context = "\n\n".join(relevant_chunks)

            prompt = f"""
You are Micro Bot, a friendly and helpful AI assistant.

Use the document context below to answer the user's question.
Only use the context if it is relevant.
If the answer is not in the context, say that clearly.

Document Context:
{context}

User Question:
{user_message}
"""
        elif document_text.strip():
            prompt = f"""
You are Micro Bot, a friendly and helpful AI assistant.

A document was uploaded, but no relevant chunks were found.
If needed, answer generally and clearly mention that no direct document match was found.

User Question:
{user_message}
"""
            used_indices = []
            relevant_chunks = []
        else:
            prompt = user_message
            used_indices = []
            relevant_chunks = []

        response = ollama.chat(
            model="qwen2.5:0.5b",
            messages=[
                {
                    "role": "system",
                    "content": "You are Micro Bot, a friendly, helpful AI assistant. Keep answers clear and simple."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        bot_response = response["message"]["content"]

        return jsonify({
            "response": bot_response,
            "sources": [
                {
                    "chunk_index": used_indices[i],
                    "preview": relevant_chunks[i][:220] + ("..." if len(relevant_chunks[i]) > 220 else "")
                }
                for i in range(len(relevant_chunks))
            ],
            "uploaded_filename": uploaded_filename
        })

    except Exception as e:
        return jsonify({
            "response": f"Local model error: {str(e)}",
            "sources": [],
            "uploaded_filename": uploaded_filename
        }), 500

@app.route("/upload", methods=["POST"])
def upload_file():
    global document_text, document_chunks, document_embeddings, faiss_index, uploaded_filename

    if "file" not in request.files:
        return jsonify({"response": "No file was uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"response": "No file selected."}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    uploaded_filename = file.filename

    if file.filename.lower().endswith(".pdf"):
        try:
            reader = PdfReader(file_path)
            extracted_text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"

            document_text = extracted_text
            document_chunks = split_text_into_chunks(document_text)

            if document_chunks:
                document_embeddings = embedding_model.encode(document_chunks).tolist()
                faiss_index = create_faiss_index(document_embeddings)
            else:
                document_embeddings = []
                faiss_index = None

            if document_text.strip():
                return jsonify({
                    "response": f"PDF uploaded, split into {len(document_chunks)} chunk(s), embeddings created, and FAISS index built successfully: {file.filename}",
                    "filename": uploaded_filename
                })
            else:
                return jsonify({
                    "response": f"PDF uploaded, but no readable text was found in: {file.filename}",
                    "filename": uploaded_filename
                })

        except Exception as e:
            return jsonify({
                "response": f"PDF uploaded, but processing failed: {str(e)}",
                "filename": uploaded_filename
            }), 500

    else:
        document_text = ""
        document_chunks = []
        document_embeddings = []
        faiss_index = None
        return jsonify({
            "response": f"File uploaded successfully: {file.filename}. Only PDF reading is enabled right now.",
            "filename": uploaded_filename
        })

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "uploaded_filename": uploaded_filename,
        "chunk_count": len(document_chunks)
    })

if __name__ == "__main__":
    app.run(debug=True)
    if __name__ == "__main__":
    app.run(debug=True)