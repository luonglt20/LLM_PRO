import os
import json
import urllib.request
import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from pypdf import PdfReader

# 1. Define the Agent State
class AgentState(TypedDict):
    query: str
    api_key: str
    papers: List[Dict[str, Any]]
    agent_logs: List[Dict[str, Any]]
    retrieved_contexts: List[Dict[str, Any]]
    final_report: str

# 2. General LLM Calling Helper
def call_gemini(prompt: str, api_key: str, model: str = "gemini-1.5-flash") -> str:
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
        with urllib.request.urlopen(req, timeout=20) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            return res_data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print(f"Error calling Gemini in agent: {e}")
        return f"Error: Failed to fetch response from Gemini. Details: {e}"

# 3. Agent 1: ArxivSearchAgent
def search_arxiv_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    logs = list(state.get("agent_logs", []))
    
    logs.append({
        "agent": "ArxivSearchAgent",
        "message": f"Analyzing query keywords and searching arXiv database for: '{query}'..."
    })
    
    # Load papers from local database papers.json as our searchable catalog
    papers_path = os.path.join(os.path.dirname(__file__), "papers.json")
    matched_papers = []
    
    if os.path.exists(papers_path):
        try:
            with open(papers_path, "r") as f:
                all_papers = json.load(f)
                
            # Perform word boundary matching on keywords
            query_words = [w.lower() for w in query.split() if len(w) > 1]
            if not query_words:
                query_words = [query.lower()]
                
            scored_papers = []
            for paper in all_papers:
                text = (paper["title"] + " " + paper["abstract"]).lower()
                matches = sum(1 for qw in query_words if re.search(r'\b' + re.escape(qw), text))
                if matches > 0:
                    scored_papers.append((paper, matches))
                    
            # Sort by match count descending, then relevance score descending
            scored_papers = sorted(scored_papers, key=lambda x: (x[1], x[0].get("relevance_score", 0.0)), reverse=True)
            matched_papers = [p[0] for p in scored_papers[:3]]
            
        except Exception as e:
            print(f"Error loading papers in SearchAgent: {e}")
            
    # Fallback to top 2 papers if no matches found
    if not matched_papers and os.path.exists(papers_path):
        try:
            with open(papers_path, "r") as f:
                all_papers = json.load(f)
                matched_papers = all_papers[:2]
        except Exception:
            pass
            
    logs.append({
        "agent": "ArxivSearchAgent",
        "message": f"Found {len(matched_papers)} relevant papers for research topic."
    })
    
    return {
        "papers": matched_papers,
        "agent_logs": logs
    }

# 4. Agent 2: PaperCriticAgent
def critic_papers_node(state: AgentState) -> Dict[str, Any]:
    papers = state["papers"]
    api_key = state["api_key"]
    logs = list(state.get("agent_logs", []))
    retrieved = []
    
    if not papers:
        logs.append({
            "agent": "PaperCriticAgent",
            "message": "No papers provided. Skipping analysis."
        })
        return {"agent_logs": logs, "retrieved_contexts": []}
        
    for p in papers:
        logs.append({
            "agent": "PaperCriticAgent",
            "message": f"Analyzing methodology, constraints, and results for: '{p['title']}'..."
        })
        
        # Check if RAG chunks are cached
        pdf_cache_dir = os.path.join(os.path.dirname(__file__), "pdf_cache")
        cache_file = os.path.join(pdf_cache_dir, f"{p['id']}.json")
        summary_text = ""
        
        # Try loading text chunks to find actual paragraphs
        chunks = []
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    chunks = json.load(f)
            except Exception:
                pass
                
        if chunks:
            # Join first 4 chunks to get a good context (around 6k chars)
            summary_text = " ".join([c["text"] for c in chunks[:4]])
        else:
            summary_text = p["abstract"]
            
        if api_key:
            prompt = f"""
You are a peer-review expert. Analyze the following scientific paper context.
Summarize:
1. The main methodology and equations used.
2. The dataset, training parameters, and benchmarks.
3. The core limitations or weaknesses.

Paper Context:
{summary_text[:3000]}

Write a concise critique in 3 bullet points.
"""
            critique = call_gemini(prompt, api_key)
        else:
            # Local Heuristic Critique Fallback
            critique = f"- **Methodology**: Uses the core highlights: {p.get('highlight', 'arXiv methods')}.\n"
            critique += f"- **Ablations**: Evaluated on Category: {p['category_name']}.\n"
            critique += f"- **Limits**: Dependent on the general domain parameters defined in their abstract."
            
        retrieved.append({
            "id": p["id"],
            "title": p["title"],
            "critique": critique
        })
        
    logs.append({
        "agent": "PaperCriticAgent",
        "message": "Completed analysis of all retrieved papers."
    })
    
    return {
        "agent_logs": logs,
        "retrieved_contexts": retrieved
    }

# 5. Agent 3: LiteratureReviewAgent
def synthesize_report_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    retrieved = state["retrieved_contexts"]
    api_key = state["api_key"]
    logs = list(state.get("agent_logs", []))
    
    logs.append({
        "agent": "LiteratureReviewAgent",
        "message": "Synthesizing individual critiques to compile a complete Literature Review report..."
    })
    
    if not retrieved:
        report = "### Literature Review Fallback\n\nNo papers were retrieved to synthesize. Please try a different query."
        logs.append({
            "agent": "LiteratureReviewAgent",
            "message": "Failed to compile report: No contexts retrieved."
        })
        return {"agent_logs": logs, "final_report": report}
        
    critiques_str = ""
    for r in retrieved:
        critiques_str += f"\nPaper: {r['title']}\nAnalysis:\n{r['critique']}\n"
        
    if api_key:
        prompt = f"""
You are a principal investigator compiling a Literature Review.
Based on the individual paper reviews provided below, write a synthesis report about the research topic: '{query}'.

Structure:
# Literature Review: {query}
## 1. Introduction
## 2. Methodology Comparisons (Group papers by their approaches)
## 3. Key Scientific Constraints & Gaps
## 4. Proposed Future Directions

Individual Paper Reviews:
{critiques_str}

Use professional scientific language and format in clean Markdown.
"""
        report = call_gemini(prompt, api_key)
    else:
        # Local Heuristic Synthesis
        report = f"# Literature Review: {query}\n\n"
        report += "*(Local Heuristic Report - Gemini API Key is unconfigured)*\n\n"
        report += "## 1. Introduction\n"
        report += f"This report synthesizes the latest papers regarding the topic of '{query}'.\n\n"
        report += "## 2. Methodology Comparisons\n"
        for r in retrieved:
            report += f"### {r['title']}\n"
            report += f"{r['critique']}\n\n"
        report += "## 3. Key Scientific Gaps & Future Directions\n"
        report += "- There is a need to test these methodologies on larger datasets.\n"
        report += "- Integrating multi-agent architectures could further optimize automated workflows.\n"
        
    logs.append({
        "agent": "LiteratureReviewAgent",
        "message": "Literature Review report compiled successfully."
    })
    
    return {
        "agent_logs": logs,
        "final_report": report
    }

# 6. Build the LangGraph Workflow State Machine
def run_agent_workflow(query: str, api_key: str) -> Dict[str, Any]:
    # Initialize StateGraph with our state shape schema
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
    
    # Run Graph execution synchronously
    initial_state = {
        "query": query,
        "api_key": api_key,
        "papers": [],
        "agent_logs": [],
        "retrieved_contexts": [],
        "final_report": ""
    }
    
    print(f"Executing Multi-Agent workflow for query: '{query}'...")
    final_output = app.invoke(initial_state)
    return final_output
