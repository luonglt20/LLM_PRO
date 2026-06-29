import os
import json
import random
import numpy as np
import urllib.request
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from pypdf import PdfReader

from data_loader import load_or_process_papers, DATA_FILE
from recommender import train_recommender
from llm_helper import explain_recommendation, answer_pdf_question

app = Flask(__name__)
CORS(app) # Enable CORS for frontend integration

RATINGS_FILE = os.path.join(os.path.dirname(__file__), "ratings.json")
COLLECTIONS_FILE = os.path.join(os.path.dirname(__file__), "collections.json")

# Global in-memory variables
papers_db = []
ratings_db = {}
collections_db = {}
tfidf_vectorizer = None
tfidf_matrix = None

def init_db():
    global papers_db, ratings_db, collections_db, tfidf_vectorizer, tfidf_matrix
    
    # 1. Load papers (crawls arXiv if papers.json doesn't exist)
    papers_db = load_or_process_papers()
    
    # 2. Load ratings
    if os.path.exists(RATINGS_FILE):
        try:
            with open(RATINGS_FILE, "r") as f:
                ratings_db = json.load(f)
        except Exception:
            ratings_db = {}
    else:
        ratings_db = {}
        
    # 3. Load collections
    if os.path.exists(COLLECTIONS_FILE):
        try:
            with open(COLLECTIONS_FILE, "r") as f:
                collections_db = json.load(f)
        except Exception:
            collections_db = {"My Library": []}
    else:
        collections_db = {"My Library": []}
        
    # 4. Fit Global TF-IDF Vectorizer for semantic search and collection expansions
    corpus = [f"{p['title']} [SEP] {p['abstract']}" for p in papers_db]
    tfidf_vectorizer = TfidfVectorizer(max_features=256, stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    # 5. Train recommender on startup if ratings exist
    if ratings_db:
        print("Training recommender on startup...")
        papers_db = train_recommender(papers_db, ratings_db)
        save_papers_to_disk()

def save_papers_to_disk():
    global papers_db
    with open(DATA_FILE, "w") as f:
        json.dump(papers_db, f, indent=2)

def save_ratings():
    global ratings_db
    with open(RATINGS_FILE, "w") as f:
        json.dump(ratings_db, f, indent=2)

def save_collections():
    global collections_db
    with open(COLLECTIONS_FILE, "w") as f:
        json.dump(collections_db, f, indent=2)

@app.route("/api/papers", methods=["GET"])
def get_papers():
    return jsonify({
        "papers": papers_db,
        "ratings": ratings_db
    })

@app.route("/api/rate", methods=["POST"])
def rate_paper():
    global papers_db, ratings_db
    data = request.json
    paper_id = data.get("paper_id")
    rating = data.get("rating") # 1, -1, or 0 (to clear)
    
    if not paper_id:
        return jsonify({"error": "Missing paper_id"}), 400
        
    # Update ratings
    if rating == 0:
        if paper_id in ratings_db:
            del ratings_db[paper_id]
    else:
        ratings_db[paper_id] = rating
        
    save_ratings()
    
    # Retrain recommender
    papers_db = train_recommender(papers_db, ratings_db)
    save_papers_to_disk()
    
    return jsonify({
        "status": "success",
        "papers": papers_db,
        "ratings": ratings_db
    })

@app.route("/api/digest", methods=["GET"])
def get_digest():
    digest_range = request.args.get("range", "day")
    
    # We simulate daily vs weekly digests by slicing papers.
    # Daily: highest relevance papers from the last few dates.
    # Since they are pre-sorted by date in arXiv feed, we can take recent papers
    # and sort them by relevance score.
    sorted_papers = sorted(papers_db, key=lambda p: p["published"], reverse=True)
    
    if digest_range == "day":
        # Simulate last 30 papers (recent)
        candidates = sorted_papers[:40]
    else:
        # Simulate last 100 papers (weekly)
        candidates = sorted_papers[:120]
        
    # Sort candidates by relevance score
    digest_papers = sorted(candidates, key=lambda p: p["relevance_score"], reverse=True)
    
    return jsonify({
        "papers": digest_papers[:15] # Return top 15 recommendations for the digest
    })

@app.route("/api/search", methods=["POST"])
def search_papers():
    global tfidf_vectorizer, tfidf_matrix
    query = request.json.get("query", "")
    
    if not query.strip():
        # Return top relevance papers by default
        results = sorted(papers_db, key=lambda p: p["relevance_score"], reverse=True)
        return jsonify({"results": results[:20]})
        
    # Compute query vector
    query_vec = tfidf_vectorizer.transform([query])
    
    # Compute cosine similarities
    similarities = np.dot(tfidf_matrix.toarray(), query_vec.T.toarray()).flatten()
    
    # Combine TF-IDF similarity with personal relevance
    search_results = []
    for idx, paper in enumerate(papers_db):
        sim = float(similarities[idx])
        # Score combines similarity (primary) and user relevance (secondary)
        combined_score = sim * 100.0 + (paper["relevance_score"] * 0.1)
        
        search_results.append({
            "paper": paper,
            "similarity": sim,
            "combined_score": combined_score
        })
        
    # Sort by combined score descending
    search_results = sorted(search_results, key=lambda x: x["combined_score"], reverse=True)
    
    # Format return
    results = [item["paper"] for item in search_results if item["similarity"] > 0]
    
    # If no keywords matched, do a basic text search fallback using word boundaries
    if not results:
        import re
        query_words = [w.lower() for w in query.split() if w]
        for paper in papers_db:
            title_abs = (paper["title"] + " " + paper["abstract"]).lower()
            # Match only if all query words match word boundaries (prefixes) in the text
            match = True
            for qw in query_words:
                if not re.search(r'\b' + re.escape(qw), title_abs):
                    match = False
                    break
            if match:
                results.append(paper)
                
    return jsonify({"results": results[:20]})

@app.route("/api/active-learning", methods=["GET"])
def get_active_learning():
    """
    Stratified sampling with K-Means diversity clustering:
    Filters the top 20 unrated papers nearest the decision boundary, clusters their t-SNE coordinates
    into 5 distinct groups, and selects the paper closest to the centroid of each group.
    """
    unrated = [p for p in papers_db if p["id"] not in ratings_db]
    if not unrated:
        return jsonify({"papers": []})
        
    boundary_papers = sorted(unrated, key=lambda p: abs(p["relevance_score"]))
    if len(boundary_papers) <= 5:
        return jsonify({"papers": boundary_papers})
        
    # Apply K-Means clustering on t-SNE coordinates for the top 20 papers
    candidates = boundary_papers[:20]
    coords = np.array([[p["x"], p["y"]] for p in candidates])
    
    try:
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        
        selected_papers = []
        for cluster_idx in range(5):
            cluster_mask = (labels == cluster_idx)
            cluster_candidates = [candidates[i] for i, m in enumerate(cluster_mask) if m]
            if cluster_candidates:
                centroid = kmeans.cluster_centers_[cluster_idx]
                # Sort by squared distance to centroid
                cluster_candidates = sorted(cluster_candidates, key=lambda p: (p["x"] - centroid[0])**2 + (p["y"] - centroid[1])**2)
                selected_papers.append(cluster_candidates[0])
                
        # Fallback padding if we failed to fill 5 clusters
        for p in boundary_papers:
            if len(selected_papers) >= 5:
                break
            if p not in selected_papers:
                selected_papers.append(p)
    except Exception as e:
        print(f"K-Means clustering failed: {e}. Falling back to standard boundary papers.")
        selected_papers = boundary_papers[:5]
        
    return jsonify({
        "papers": selected_papers[:5]
    })

@app.route("/api/chat-pdf", methods=["POST"])
def chat_pdf():
    data = request.json
    paper_id = data.get("paper_id")
    question = data.get("question")
    api_key = data.get("api_key")
    
    if not paper_id or not question:
        return jsonify({"error": "Missing paper_id or question"}), 400
        
    # PDF Cache directory Setup
    pdf_cache_dir = os.path.join(os.path.dirname(__file__), "pdf_cache")
    os.makedirs(pdf_cache_dir, exist_ok=True)
    cache_file = os.path.join(pdf_cache_dir, f"{paper_id}.json")
    
    chunks = []
    
    # 1. Load from cache or download and split PDF
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                chunks = json.load(f)
        except Exception as e:
            print(f"Failed to read cache: {e}")
            
    if not chunks:
        temp_pdf_path = os.path.join(pdf_cache_dir, f"{paper_id}.pdf")
        arxiv_pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        
        # Download paper PDF
        try:
            print(f"Downloading PDF from {arxiv_pdf_url}...")
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(arxiv_pdf_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response, open(temp_pdf_path, "wb") as out_file:
                out_file.write(response.read())
        except Exception as e:
            return jsonify({"error": f"Failed to download PDF from arXiv: {e}"}), 500
            
        # Parse PDF text page-by-page (Hierarchical Parent-Child Chunking)
        try:
            reader = PdfReader(temp_pdf_path)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                
                # 1. Create Parent chunks (size 1500, overlap 300)
                parent_limit = 1500
                parent_overlap = 300
                parent_start = 0
                parents = []
                while parent_start < len(text):
                    parent_end = parent_start + parent_limit
                    parents.append(text[parent_start:parent_end])
                    parent_start += (parent_limit - parent_overlap)
                
                # 2. Create Child chunks (size 300, overlap 50) and link them to parent context
                child_limit = 300
                child_overlap = 50
                child_start = 0
                while child_start < len(text):
                    child_end = child_start + child_limit
                    child_text = text[child_start:child_end]
                    
                    # Find which parent chunk contains this child chunk
                    parent_text = ""
                    for p in parents:
                        if child_text in p or p in child_text:
                            parent_text = p
                            break
                    if not parent_text and parents:
                        parent_text = parents[0]
                        
                    chunks.append({
                        "child_text": child_text,
                        "text": parent_text if parent_text else child_text, # Parent context sent to LLM
                        "page_idx": page_idx + 1
                    })
                    child_start += (child_limit - child_overlap)
            
            # Save parsed chunks as cache
            with open(cache_file, "w") as f:
                json.dump(chunks, f, indent=2)
                
        except Exception as e:
            return jsonify({"error": f"Failed to parse PDF text: {e}"}), 500
        finally:
            # Clean up temp file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
                
    if not chunks:
        return jsonify({"error": "No text content could be extracted from this PDF"}), 500
        
    # 2. Vectorize child chunks and question to find top matching chunks via Cosine Similarity
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        # Use child_text for high-precision search, falls back to text if child_text is missing
        corpus = [c.get("child_text", c["text"]) for c in chunks]
        vectorizer.fit(corpus + [question])
        
        chunk_vectors = vectorizer.transform(corpus)
        question_vector = vectorizer.transform([question])
        
        # Compute Cosine Similarity
        similarities = np.dot(chunk_vectors.toarray(), question_vector.T.toarray()).flatten()
        
        # Rank chunks
        for idx, sim in enumerate(similarities):
            chunks[idx]["score"] = float(sim)
            
        ranked_chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)
        top_chunks = ranked_chunks[:3]
    except Exception as e:
        print(f"TF-IDF chunk similarity failed: {e}. Defaulting to first 3 page chunks.")
        top_chunks = chunks[:3]
        for c in top_chunks:
            c["score"] = 0.0
            
    # 3. Ask Gemini/Heuristic to answer the question based on context chunks
    answer = answer_pdf_question(top_chunks, question, api_key)
    
    return jsonify({
        "answer": answer,
        "retrieved_chunks": top_chunks
    })

@app.route("/api/explain", methods=["POST"])
def explain_paper():
    data = request.json
    paper_id = data.get("paper_id")
    api_key = data.get("api_key") # Can be passed dynamically from frontend
    
    if not paper_id:
        return jsonify({"error": "Missing paper_id"}), 400
        
    # Find target paper
    target_paper = next((p for p in papers_db if p["id"] == paper_id), None)
    if not target_paper:
        return jsonify({"error": "Paper not found"}), 404
        
    # Gather titles of upvoted papers
    upvoted_titles = []
    for pid, rating in ratings_db.items():
        if rating == 1:
            upvoted_p = next((p for p in papers_db if p["id"] == pid), None)
            if upvoted_p:
                upvoted_titles.append(upvoted_p["title"])
                
    # Call helper
    analysis = explain_recommendation(
        upvoted_titles,
        target_paper["title"],
        target_paper["abstract"],
        target_paper["category_name"],
        api_key=api_key
    )
    
    return jsonify(analysis)

@app.route("/api/collections", methods=["GET", "POST"])
def manage_collections():
    global collections_db
    
    if request.method == "GET":
        return jsonify({"collections": collections_db})
        
    data = request.json
    action = data.get("action") # "create", "add", "remove"
    name = data.get("name")
    paper_id = data.get("paper_id")
    
    if not name:
        return jsonify({"error": "Missing collection name"}), 400
        
    if action == "create":
        if name not in collections_db:
            collections_db[name] = []
            save_collections()
            
    elif action == "add":
        if name in collections_db and paper_id and paper_id not in collections_db[name]:
            collections_db[name].append(paper_id)
            save_collections()
            
    elif action == "remove":
        if name in collections_db and paper_id and paper_id in collections_db[name]:
            collections_db[name].remove(paper_id)
            save_collections()
            
    return jsonify({
        "status": "success",
        "collections": collections_db
    })

@app.route("/api/collections/<name>/recommendations", methods=["GET"])
def get_collection_recommendations(name):
    global tfidf_matrix, tfidf_vectorizer
    
    if name not in collections_db or not collections_db[name]:
        return jsonify({"papers": []})
        
    # Get paper IDs in collection
    col_paper_ids = set(collections_db[name])
    
    # Find indices of collection papers in global papers list
    indices = [idx for idx, p in enumerate(papers_db) if p["id"] in col_paper_ids]
    
    if not indices:
        return jsonify({"papers": []})
        
    # Average embedding of collection papers
    col_embeddings = tfidf_matrix.toarray()[indices]
    avg_embedding = np.mean(col_embeddings, axis=0)
    
    # Compute similarity against all papers
    similarities = np.dot(tfidf_matrix.toarray(), avg_embedding).flatten()
    
    recommendations = []
    for idx, paper in enumerate(papers_db):
        if paper["id"] in col_paper_ids:
            continue # Skip papers already in collection
            
        recommendations.append({
            "paper": paper,
            "similarity": float(similarities[idx])
        })
        
    # Sort by similarity descending
    recommendations = sorted(recommendations, key=lambda x: x["similarity"], reverse=True)
    
    return jsonify({
        "papers": [item["paper"] for item in recommendations[:5]]
    })

@app.route("/api/conference", methods=["GET"])
def get_conference_planner():
    """
    Generates a set of mock conference poster sessions.
    Ranks them based on the average relevance of papers in each session.
    """
    # Define 4 mock poster sessions
    sessions = [
        {
            "id": "session_1",
            "name": "Poster Session A: Deep Learning Foundations & Optimization",
            "time": "Day 1, 09:00 - 11:00",
            "keywords": ["gradient", "loss", "network", "train", "optimization", "transformer", "attention"]
        },
        {
            "id": "session_2",
            "name": "Poster Session B: Computer Vision & Generative Models",
            "time": "Day 1, 14:00 - 16:00",
            "keywords": ["image", "object", "segmentation", "detection", "diffusion", "gan", "cnn", "vision"]
        },
        {
            "id": "session_3",
            "name": "Poster Session C: Natural Language & Agentic LLMs",
            "time": "Day 2, 09:00 - 11:00",
            "keywords": ["text", "language", "nlp", "translation", "llm", "agent", "prompt", "reasoning"]
        },
        {
            "id": "session_4",
            "name": "Poster Session D: Reinforcement Learning & Controls",
            "time": "Day 2, 14:00 - 16:00",
            "keywords": ["policy", "agent", "reinforcement", "control", "rl", "robot", "action"]
        }
    ]
    
    # Distribute papers to sessions based on keywords matching
    session_papers = {s["id"]: [] for s in sessions}
    
    for paper in papers_db:
        content = (paper["title"] + " " + paper["abstract"]).lower()
        
        # Calculate matching scores for each session
        scores = {}
        for s in sessions:
            score = sum(1 for kw in s["keywords"] if kw in content)
            scores[s["id"]] = score
            
        # Assign to best matching session, or fallback based on primary category
        best_session_id = max(scores, key=scores.get)
        if scores[best_session_id] > 0:
            session_papers[best_session_id].append(paper)
        else:
            # Fallback mapping
            cat = paper["primary_category"]
            if cat == "cs.CV":
                session_papers["session_2"].append(paper)
            elif cat == "cs.CL":
                session_papers["session_3"].append(paper)
            elif cat == "cs.LG":
                session_papers["session_1"].append(paper)
            else:
                session_papers["session_4"].append(paper)
                
    # Rank sessions by average paper relevance score
    ranked_sessions = []
    for s in sessions:
        papers_in_s = session_papers[s["id"]]
        avg_relevance = 0.0
        if papers_in_s:
            avg_relevance = float(np.mean([p["relevance_score"] for p in papers_in_s]))
            
        # Sort papers in session by relevance score descending
        sorted_papers = sorted(papers_in_s, key=lambda p: p["relevance_score"], reverse=True)
        
        ranked_sessions.append({
            "id": s["id"],
            "name": s["name"],
            "time": s["time"],
            "avg_relevance": float(np.round(avg_relevance, 2)),
            "papers": sorted_papers
        })
        
    # Sort sessions by average relevance descending
    ranked_sessions = sorted(ranked_sessions, key=lambda s: s["avg_relevance"], reverse=True)
    
    return jsonify({
        "sessions": ranked_sessions
    })

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
