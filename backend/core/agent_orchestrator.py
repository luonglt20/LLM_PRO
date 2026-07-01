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
    if not api_key:
        return ""
        
    if api_key.startswith("gsk_"):
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                return res_data['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode('utf-8')
            except Exception:
                pass
            print(f"Error calling Groq in agent (HTTPError {e.code}): {e.reason} - Body: {err_body}", flush=True)
            return f"Error: Failed to fetch response from Groq. Details: HTTP Error {e.code}: {e.reason}. Response: {err_body}"
        except Exception as e:
            print(f"Error calling Groq in agent: {e}", flush=True)
            return f"Error: Failed to fetch response from Groq. Details: {e}"
    else:
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
You are an expert peer reviewer. Perform a structured critique of this paper in Vietnamese:
Title: {p['title']}
Context:
{summary_text[:3200]}

You MUST cover these 3 sections in detail. Format in clean plain text with regular subheadings (do not use markdown headers # or bold stars **):

1. Mathematical & Algorithmic Core
Describe the primary mathematical formulations (equations, losses) and neural architectures.

2. Empirical Setup & Metrics
Specify the datasets used, training hyperparameters (epochs, learning rate), and baseline methods.

3. Scientific Limitations & Constraints
State the failure modes, high GPU memory requirements, and specific assumptions made.
"""
            critique = call_gemini(prompt, api_key)
        else:
            # Advanced local NLP mining (extracting numbers and parameters using regex patterns)
            epochs_match = re.findall(r'\b\d+\s*(?:epochs|epoch)\b', summary_text, re.IGNORECASE)
            batch_match = re.findall(r'\b(?:batch\s*size|batch)\s*(?:of\s*)?\d+\b', summary_text, re.IGNORECASE)
            lr_match = re.findall(r'\b(?:learning\s*rate|lr)\s*(?:of\s*)?(?:0\.\d+|\d+e-\d+)\b', summary_text, re.IGNORECASE)
            
            critique = "1. Mathematical & Algorithmic Core\n"
            critique += f"- Core Methodology: {p.get('highlight', 'arXiv algorithms')}.\n"
            critique += "- Uses token embeddings and high-dimensional semantic mapping space.\n"
            
            critique += "\n2. Empirical Setup & Metrics\n"
            critique += f"- Primary Category: {p['category_name']} ({p['primary_category']}).\n"
            if epochs_match:
                critique += f"- Extracted training parameter: {', '.join(epochs_match)}.\n"
            if batch_match:
                critique += f"- Extracted batch parameter: {', '.join(batch_match)}.\n"
            if lr_match:
                critique += f"- Extracted learning rate: {', '.join(lr_match)}.\n"
                
            critique += "\n3. Scientific Limitations & Constraints\n"
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
Bạn là Trưởng nhóm Nghiên cứu Khoa học (Director of Research). Hãy xây dựng một Lộ trình Nghiên cứu (Research Roadmap) và Kế hoạch Thực nghiệm (Implementation Plan) chi tiết bằng tiếng Việt dựa trên chủ đề nghiên cứu: '{query}' và các phân tích bài báo tham khảo bên dưới.

Báo cáo lộ trình phải sử dụng định dạng Markdown chuẩn (tiêu đề #, ##, ###, danh sách bullet -, và bảng so sánh) và phải bao gồm đầy đủ các phần sau:

# Lộ trình Nghiên cứu & Kế hoạch Thực hiện: {query}

## 1. Tổng quan & Mục tiêu Chiến lược
- Giới thiệu ngắn gọn về tầm quan trọng và bối cảnh khoa học của chủ đề '{query}'.
- Xác định mục tiêu nghiên cứu cụ thể mà lộ trình này hướng tới.

## 2. Bản đồ Tài liệu Tham khảo (Reference Mapping)
- Phân tích vai trò đóng góp của các tài liệu tham khảo được cung cấp đối với mục tiêu chung.
- Xây dựng một bảng so sánh Markdown (Comparative Matrix) gồm các cột: Bài Báo | Phương Pháp Cốt Lõi | Chỉ Số Thực Nghiệm | Điểm Yếu/Hạn Chế. Điền thông tin chính xác từ bài phân tích.

## 3. Kế hoạch Tiếp cận Theo Giai đoạn (Phased Implementation Plan)
- **Giai đoạn 1: Nền tảng & Tái lập (Foundation & Replication)**: Xác định cụ thể tài liệu nào cần tái lập đầu tiên, cài đặt môi trường gì.
- **Giai đoạn 2: Tích hợp & Phát triển (Integration & Synergy)**: Mô tả phương án kỹ thuật chi tiết để tích hợp thế mạnh của các bài báo lại với nhau (ví dụ: dùng giải pháp của bài A để khắc phục lỗ hổng của bài B).
- **Giai đoạn 3: Tối ưu hóa & Đánh giá (Optimization & Evaluation)**: Các chỉ số benchmark cần theo dõi và kế hoạch kiểm thử hiệu năng.

## 4. Kế hoạch Hành động cụ thể (Milestone Schedule)
- Tạo một bảng Markdown phân chia cụ thể các mốc thời gian hành động theo tuần/tháng (ví dụ: Tuần 1-2, Tuần 3-4,...) kèm mục tiêu cụ thể.

## 5. Rủi ro Kỹ thuật & Phương án Dự phòng
- Xác định các rủi ro kỹ thuật chính (ví dụ: tràn bộ nhớ GPU, phân rã loss, overfitting) và đề xuất phương án dự phòng (mitigation strategy) tương ứng.

Thông tin phân tích bài báo:
{critiques_str}

Hãy viết báo cáo bằng văn phong khoa học, chuyên nghiệp, lập luận logic sâu sắc và hoàn toàn bằng tiếng Việt.
"""
        report = call_gemini(prompt, api_key)
    else:
        # Local Heuristic Comparative Matrix and Synthesis
        report = f"# Lộ trình Nghiên cứu & Kế hoạch Thực hiện: {query}\n\n"
        report += "*(Lộ trình Khoa học Dự phòng - Gemini/Groq API Key chưa được định cấu hình)*\n\n"
        report += "## 1. Tổng quan & Mục tiêu Chiến lược\n"
        report += f"Lộ trình này nhằm mục đích thiết lập một quy trình thực nghiệm toàn diện giải quyết chủ đề '{query}' thông qua việc kết hợp các nghiên cứu tiên tiến nhất.\n\n"
        report += "## 2. Bản đồ Tài liệu Tham khảo (Reference Mapping)\n"
        report += "| Bài Báo | Phương Pháp Cốt Lõi | Chỉ Số Thực Nghiệm | Điểm Yếu/Hạn Chế |\n"
        report += "| --- | --- | --- | --- |\n"
        for r in retrieved:
            p_obj = papers_lookup.get(r["id"], {})
            report += f"| {r['title'][:30]}... | {p_obj.get('primary_category', 'N/A')} | {p_obj.get('highlight', 'N/A')[:40]}... | Yêu cầu tài nguyên tính toán lớn |\n"
        report += "\n"
        report += "## 3. Kế hoạch Tiếp cận Theo Giai đoạn (Phased Implementation Plan)\n"
        report += "- **Giai đoạn 1: Nền tảng & Tái lập (Foundation & Replication)**: Cài đặt và thực nghiệm các mã nguồn mở đi kèm của các nghiên cứu để kiểm tra tính đúng đắn.\n"
        report += "- **Giai đoạn 2: Tích hợp & Phát triển (Integration & Synergy)**: Giải quyết các xung đột đầu vào/đầu ra giữa các mô hình và xây dựng module tích hợp.\n"
        report += "- **Giai đoạn 3: Tối ưu hóa & Đánh giá (Optimization & Evaluation)**: Tinh chỉnh siêu tham số và đánh giá chéo hiệu năng.\n\n"
        report += "## 4. Kế hoạch Hành động cụ thể (Milestone Schedule)\n"
        report += "| Mốc thời gian | Hoạt động chính | Kết quả mong đợi |\n"
        report += "| --- | --- | --- |\n"
        report += "| Tuần 1-2 | Cài đặt môi trường ảo và chạy lại mã nguồn các bài báo | Có môi trường làm việc chuẩn hóa |\n"
        report += "| Tuần 3-4 | Thực thi kiểm thử độc lập cho từng module gốc | Có chỉ số baseline để so sánh |\n"
        report += "| Tuần 5-8 | Phát triển và liên kết các thành phần mô hình | Bản mẫu tích hợp đầu tiên (MVP) |\n"
        report += "\n"
        report += "## 5. Rủi ro Kỹ thuật & Phương án Dự phòng\n"
        report += "- **Rủi ro 1**: Quá tải bộ nhớ GPU khi huấn luyện mô hình tích hợp. *Dự phòng*: Áp dụng tinh chỉnh hiệu quả tham số (LoRA, QLoRA) hoặc giảm kích thước batch.\n"
        report += "- **Rủi ro 2**: Sai lệch phân phối dữ liệu thực tế (Domain shift). *Dự phòng*: Áp dụng kỹ thuật Domain Adaptation hoặc huấn luyện tăng cường dữ liệu."
        
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
