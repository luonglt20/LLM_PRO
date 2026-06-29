import sys
import os
import numpy as np

# Adjust path to import backend modules
sys.path.append(os.path.dirname(__file__))

from llm_helper import answer_pdf_question

def test_rag():
    print("=== Testing PDF RAG Engine ===")
    
    # 1. Mocking chunks
    chunks = [
        {"text": "The Transformer architecture uses multi-head self-attention mechanisms to map query and key-value vectors.", "page_idx": 1},
        {"text": "Our model achieved a test error rate of 1.2 percent on standard benchmark datasets like ImageNet.", "page_idx": 4},
        {"text": "For training, we configure a learning rate of 0.001 with Adam optimization and batch size 64.", "page_idx": 5}
    ]
    
    # 2. Test Cosine Similarity matching manually (Simulating TF-IDF matching in RAG route)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    question = "How is the learning rate configured for Adam training?"
    
    corpus = [c["text"] for c in chunks]
    vectorizer.fit(corpus + [question])
    
    chunk_vectors = vectorizer.transform(corpus)
    question_vector = vectorizer.transform([question])
    
    similarities = np.dot(chunk_vectors.toarray(), question_vector.T.toarray()).flatten()
    
    # We expect chunk 3 (learning rate and Adam) to have the highest similarity score
    best_match_idx = np.argmax(similarities)
    print(f"Question: '{question}'")
    print(f"Similarity scores: {similarities}")
    print(f"Best match chunk: Page {chunks[best_match_idx]['page_idx']} - \"{chunks[best_match_idx]['text']}\"")
    
    assert best_match_idx == 2, "Expected chunk 3 (learning rate) to be the best match!"
    assert similarities[2] > 0.0, "Expected positive similarity score!"
    
    # 3. Test local heuristic fallback answer
    print("\nTesting RAG Heuristic Fallback (without API Key)...")
    res_heuristic = answer_pdf_question(chunks, question, api_key=None)
    print(res_heuristic)
    
    assert "[Local Index Fallback]" in res_heuristic, "Heuristic fallback indicator missing!"
    
    print("\nPDF RAG Engine tests PASSED successfully!")

if __name__ == "__main__":
    test_rag()
