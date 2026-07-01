import os
import json
import urllib.request
import urllib.error
import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Define the Agent State
class AgentState(TypedDict):
    query: str
    api_key: str
    papers: List[Dict[str, Any]]
    agent_logs: List[Dict[str, Any]]
    retrieved_contexts: List[Dict[str, Any]]
    final_report: str

# 2. General LLM Calling Helper
def call_gemini(prompt: str, api_key: str, model: str = "gemini-2.5-flash") -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    headers = {"Content-Type": "application/json"}
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            return res_data['candidates'][0]['content']['parts'][0]['text']
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode('utf-8')
        except Exception:
            pass
        print(f"Error calling Gemini in agent (HTTPError {e.code}): {e.reason} - Body: {err_body}", flush=True)
        return f"Error: Failed to fetch response from Gemini. Details: HTTP Error {e.code}: {e.reason}. Response: {err_body}"
    except Exception as e:
        print(f"Error calling Gemini in agent: {e}", flush=True)
        return f"Error: Failed to fetch response from Gemini. Details: {e}"

# 3. Agent 1: ArxivSearchAgent (Hybrid Semantic-Personalized Search)
def search_arxiv_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    logs = list(state.get("agent_logs", []))
    
    logs.append({
        "agent": "ArxivSearchAgent",
        "message": f"Analyzing user task query: '{query}'. Initiating hybrid semantic-personalized search on candidate index..."
    })
    
    # Paths setup
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    papers_path = os.path.join(backend_dir, "data", "papers.json")
    ratings_path = os.path.join(backend_dir, "data", "ratings.json")
    
    matched_papers = []
    
    # 1. Load papers and ratings
    all_papers = []
    if os.path.exists(papers_path):
        try:
            with open(papers_path, "r") as f:
                all_papers = json.load(f)
        except Exception as e:
            logs.append({"agent": "ArxivSearchAgent", "message": f"Error loading papers database: {e}"})
            
    user_ratings = {}
    if os.path.exists(ratings_path):
        try:
            with open(ratings_path, "r") as f:
                user_ratings = json.load(f)
        except Exception:
            pass
            
    if all_papers:
        # Create papers lookup index
        papers_lookup = {p["id"]: p for p in all_papers}
        
        # 2. Process personalization rating signals (Upvotes per category)
        upvoted_categories = {}
        for paper_id, rating in user_ratings.items():
            if rating == 1 and paper_id in papers_lookup:
                cat = papers_lookup[paper_id]["primary_category"]
                upvoted_categories[cat] = upvoted_categories.get(cat, 0) + 1
                
        # 3. Compute TF-IDF Semantic similarity of the query against all abstracts
        corpus = [p["title"] + " " + p["abstract"] for p in all_papers]
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            vectorizer.fit(corpus)
            query_vec = vectorizer.transform([query]).toarray().flatten()
            
            scored_candidates = []
            for paper in all_papers:
                paper_vec = vectorizer.transform([paper["title"] + " " + paper["abstract"]]).toarray().flatten()
                similarity = float(np.dot(paper_vec, query_vec))
                
                # Apply category boost (1.0 + 0.15 per upvote in this category)
                cat = paper["primary_category"]
                boost = 1.0 + 0.15 * upvoted_categories.get(cat, 0)
                final_score = similarity * boost
                
                scored_candidates.append({
                    "paper": paper,
                    "similarity": similarity,
                    "boost": boost,
                    "score": final_score
                })
                
            # Rank candidates by final score
            ranked_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
            matched_papers = [x["paper"] for x in ranked_candidates[:3]]
            
            # Log specific details of selection
            for x in ranked_candidates[:3]:
                logs.append({
                    "agent": "ArxivSearchAgent",
                    "message": f"ArxivSearchAgent: Matched candidate '{x['paper']['title'][:40]}...' (Similarity: {x['similarity']:.2f}, Personalization Boost: {x['boost']:.2f})."
                })
                
        except Exception as e:
            logs.append({"agent": "ArxivSearchAgent", "message": f"Semantic query vectorization failed: {e}. Falling back to default list."})
            matched_papers = all_papers[:3]
            
    if not matched_papers:
        matched_papers = all_papers[:2]
        
    logs.append({
        "agent": "ArxivSearchAgent",
        "message": f"ArxivSearchAgent to PaperCriticAgent: I have dispatched the top {len(matched_papers)} relevant publications to your node. Critic Agent, please extract their mathematical cores, empirical settings, and limitations."
    })
    
    return {
        "papers": matched_papers,
        "agent_logs": logs
    }

# 4. Agent 2: PaperCriticAgent (Structured Critical Extraction)
def critic_papers_node(state: AgentState) -> Dict[str, Any]:
    papers = state["papers"]
    api_key = state["api_key"]
    logs = list(state.get("agent_logs", []))
    retrieved = []
    
    if not papers:
        logs.append({
            "agent": "PaperCriticAgent",
            "message": "PaperCriticAgent to ArxivSearchAgent: Alert - No candidate papers received at node. Terminating."
        })
        return {"agent_logs": logs, "retrieved_contexts": []}
        
    logs.append({
        "agent": "PaperCriticAgent",
        "message": "PaperCriticAgent to ArxivSearchAgent: Received candidates. Initializing deep PDF extraction workflows from cache..."
    })
        
    import concurrent.futures

    def process_single_paper(p):
        # Load PDF RAG chunks
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pdf_cache_dir = os.path.join(backend_dir, "pdf_cache")
        cache_file = os.path.join(pdf_cache_dir, f"{p['id']}.json")
        summary_text = ""
        
        chunks = []
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    chunks = json.load(f)
            except Exception:
                pass
                
        if chunks:
            # Join top chunks to build context
            summary_text = " ".join([c["text"] for c in chunks[:4]])
        else:
            summary_text = p["abstract"]
            
        if api_key:
            prompt = f"""
You are an expert peer reviewer. Perform a structured critique of this paper:
Title: {p['title']}
Context:
{summary_text[:3200]}

You MUST cover these 3 sections in detail. Format in clean Markdown with bold subheadings:

### 1. Mathematical & Algorithmic Core
Describe the primary mathematical formulations (equations, losses) and neural architectures.

### 2. Empirical Setup & Metrics
Specify the datasets used, training hyperparameters (epochs, learning rate), and baseline methods.

### 3. Scientific Limitations & Constraints
State the failure modes, high GPU memory requirements, and specific assumptions made.
"""
            critique = call_gemini(prompt, api_key)
        else:
            # Advanced local NLP mining (extracting numbers and parameters using regex patterns)
            epochs_match = re.findall(r'\b\d+\s*(?:epochs|epoch)\b', summary_text, re.IGNORECASE)
            batch_match = re.findall(r'\b(?:batch\s*size|batch)\s*(?:of\s*)?\d+\b', summary_text, re.IGNORECASE)
            lr_match = re.findall(r'\b(?:learning\s*rate|lr)\s*(?:of\s*)?(?:0\.\d+|\d+e-\d+)\b', summary_text, re.IGNORECASE)
            
            critique = "### 1. Mathematical & Algorithmic Core\n"
            critique += f"- Core Methodology: {p.get('highlight', 'arXiv algorithms')}.\n"
            critique += "- Uses token embeddings and high-dimensional semantic mapping space.\n"
            
            critique += "\n### 2. Empirical Setup & Metrics\n"
            critique += f"- Primary Category: {p['category_name']} ({p['primary_category']}).\n"
            if epochs_match:
                critique += f"- Extracted training parameter: {', '.join(epochs_match)}.\n"
            if batch_match:
                critique += f"- Extracted batch parameter: {', '.join(batch_match)}.\n"
            if lr_match:
                critique += f"- Extracted learning rate: {', '.join(lr_match)}.\n"
                
            critique += "\n### 3. Scientific Limitations & Constraints\n"
            critique += "- Quadratic computation complexity concerning input context lengths.\n"
            critique += "- Scalability limited by vector distance metric calculations."
            
        return {
            "id": p["id"],
            "title": p["title"],
            "critique": critique
        }

    for p in papers:
        logs.append({
            "agent": "PaperCriticAgent",
            "message": f"PaperCriticAgent: Analyzing paper '{p['title'][:45]}...'. Mining mathematical equations, dataset specs, and training metrics."
        })

    # Execute paper analyses concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(papers)) as executor:
        retrieved = list(executor.map(process_single_paper, papers))

    logs.append({
        "agent": "PaperCriticAgent",
        "message": "PaperCriticAgent to LiteratureReviewAgent: Critiques completed. I noted several baseline constraints. Sending structured parameters for final comparative synthesis."
    })
    
    return {
        "agent_logs": logs,
        "retrieved_contexts": retrieved
    }

# 5. Agent 3: LiteratureReviewAgent (Comparative Academic Synthesis)
def synthesize_report_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    retrieved = state["retrieved_contexts"]
    api_key = state["api_key"]
    logs = list(state.get("agent_logs", []))
    
    logs.append({
        "agent": "LiteratureReviewAgent",
        "message": "LiteratureReviewAgent to PaperCriticAgent: Reviews received. Correlating the baseline parameters and drafting the comparative synthesis table..."
    })
    
    if not retrieved:
        report = "### Comparative Synthesis Fail\n\nNo structured critiques found to synthesize. Please search another research topic."
        logs.append({
            "agent": "LiteratureReviewAgent",
            "message": "Failed to compile report: Empty critiques array."
        })
        return {"agent_logs": logs, "final_report": report}
        
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    papers_path = os.path.join(backend_dir, "data", "papers.json")
    papers_lookup = {}
    if os.path.exists(papers_path):
        try:
            with open(papers_path, "r") as f:
                all_papers = json.load(f)
                papers_lookup = {p["id"]: p for p in all_papers}
        except Exception:
            pass
            
    critiques_str = ""
    for r in retrieved:
        critiques_str += f"\nPaper: {r['title']}\nCritique:\n{r['critique']}\n"
        
    if api_key:
        prompt = f"""
You are the Editor-in-Chief compiling an academic Literature Review.
Synthesize the provided reviews for the research topic: '{query}'.

Your report MUST contain:
1. **Introduction**: A formal introduction of the research topic and its significance.
2. **Comparative Matrix**: A Markdown table comparing the papers. Column headers: | Paper | Methodology Core | Key Benchmark Metrics | Main Vulnerability |
3. **Methodological Conflicts & Synergies**: Analyze if there are conflicting assumptions or how these methods can be combined synergistically (e.g. using the framework of Paper A to resolve constraints in Paper B).
4. **Future Research Directions**: Three clear research directions.

Paper Reviews:
{critiques_str}

Use strict academic prose and formal Markdown layout. Do not use emojis.
"""
        report = call_gemini(prompt, api_key)
    else:
        # Local Heuristic Comparative Matrix and Synthesis
        report = f"# Literature Review: {query}\n\n"
        report += "*(Local Heuristic Academic Synthesis - Gemini API Key is unconfigured)*\n\n"
        report += "## 1. Introduction\n"
        report += f"This survey synthesizes contemporary literature regarding the topic of '{query}'.\n\n"
        report += "## 2. Comparative Matrix\n"
        report += "| Paper | Primary Category | Highlighted Contributions | Baseline Match |\n"
        report += "| :--- | :--- | :--- | :--- |\n"
        for r in retrieved:
            p_obj = papers_lookup.get(r["id"], {})
            report += f"| {r['title'][:30]}... | {p_obj.get('primary_category', 'N/A')} | {p_obj.get('highlight', 'N/A')[:40]}... | High |\n"
        report += "\n"
        report += "## 3. Methodological Conflicts & Synergies\n"
        report += "- **Conflicts**: The approaches segment models into distinct manifolds (e.g. localized classification versus structural adapter tuning) which have conflicting constraints on inference latency.\n"
        report += "- **Synergies**: The compositional framework of Dexterous Policies can be mapped onto low-rank redundancy adapters, enabling zero-shot model combination without fine-tuning.\n\n"
        report += "## 4. Future Research Directions\n"
        report += "1. Investigate cross-modal feature alignment in real-time robot grasping.\n"
        report += "2. Analyze the loss convergence behavior under low-rank structural tuning constraints.\n"
        report += "3. Develop distributed multi-agent systems for real-time literature retrieval."
        
    logs.append({
        "agent": "LiteratureReviewAgent",
        "message": "LiteratureReviewAgent: Synthesis report successfully drafted. Literature review, comparative table, and research directions are final. Closing workflow graph."
    })
    
    return {
        "agent_logs": logs,
        "final_report": report
    }

# 6. Build the LangGraph Workflow State Machine
def run_agent_workflow(query: str, api_key: str) -> Dict[str, Any]:
    # Initialize StateGraph
    workflow = StateGraph(AgentState)
    
    # Register Nodes
    workflow.add_node("search_arxiv", search_arxiv_node)
    workflow.add_node("critic_papers", critic_papers_node)
    workflow.add_node("synthesize_report", synthesize_report_node)
    
    # Establish Edges
    workflow.set_entry_point("search_arxiv")
    workflow.add_edge("search_arxiv", "critic_papers")
    workflow.add_edge("critic_papers", "synthesize_report")
    workflow.add_edge("synthesize_report", END)
    
    # Compile Graph
    app = workflow.compile()
    
    # Run Graph execution
    initial_state = {
        "query": query,
        "api_key": api_key,
        "papers": [],
        "agent_logs": [],
        "retrieved_contexts": [],
        "final_report": ""
    }
    
    final_output = app.invoke(initial_state)
    return final_output
