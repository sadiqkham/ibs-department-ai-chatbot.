import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-120be"

@__import__("functools").lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def build_index(chunks):
    if not chunks:
        raise ValueError("No readable PDF text was found.")
    model = get_embedding_model()
    texts = [item["text"] for item in chunks]
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

def retrieve_chunks(question, index, chunks, top_k=5):
    model = get_embedding_model()
    vector = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    vector = np.asarray(vector, dtype="float32")
    count = min(top_k, len(chunks))
    scores, indices = index.search(vector, count)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = dict(chunks[int(idx)])
        item["score"] = float(score)
        results.append(item)
    return results

def answer_question(client, question, retrieved_chunks):
    if not retrieved_chunks:
        return "I couldn't find relevant information in the provided department documents. Please confirm with the IBS Department."
    context_parts = []
    for item in retrieved_chunks:
        context_parts.append(
            f"[Document: {item['document']} | Page: {item['page']}]\n{item['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    system_prompt = (
        "You are the IBS Department AI Assistant for university students. "
        "Answer questions using only the supplied document context. "
        "Do not invent teacher names, class times, rooms, policies, or other facts. "
        "If the context does not contain the answer, clearly say that you could not find it "
        "in the provided documents and recommend checking with the department. "
        "Be helpful, concise, and use simple English. When useful, mention the document name and page."
    )
    user_prompt = f"DOCUMENT CONTEXT:\n{context}\n\nSTUDENT QUESTION:\n{question}"
    completion = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return completion.choices[0].message.content or "I couldn't generate an answer. Please try again."
