import os
import re
import json
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "papers.json")

ARXIV_CATEGORIES = {
    "cs.CV": "Computer Vision",
    "cs.CL": "Natural Language Processing",
    "cs.LG": "Machine Learning",
    "cs.AI": "Artificial Intelligence",
}

def clean_text(text):
    if not text:
        return ""
    # Remove newlines and extra spaces
    text = re.sub(r"\s+", " ", text.strip())
    return text

def extract_key_sentence(abstract):
    """
    Heuristically extracts the sentence that is most related to the core contribution
    of the paper for fast skimming.
    """
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', abstract)
    
    # Priority patterns
    patterns = [
        r"\b(we propose|we introduce|we present|in this paper|in this work|our method|our approach)\b",
        r"\b(we show|we demonstrate|we develop|we design)\b",
        r"\b(shows|proposes|presents|demonstrates)\b",
        r"\b(key|contribution|novel|framework|system)\b"
    ]
    
    for pattern in patterns:
        for sent in sentences:
            if re.search(pattern, sent, re.IGNORECASE):
                return sent.strip()
                
    # Fallback to the first sentence if no pattern matches
    return sentences[0].strip() if sentences else abstract

def generate_bibtex(arxiv_url, title, authors, published):
    """
    Generates standard BibTeX format.
    """
    match = re.search(r"abs/(\d+\.\d+v\d+)", arxiv_url)
    arxiv_id = match.group(1) if match else "arxiv"
    year = published.split("-")[0] if published else "2025"
    author_list = " and ".join(authors)
    
    # Clean title for BibTeX key
    clean_title = re.sub(r"[^a-zA-Z0-9]", "", title.split()[0].lower())
    bibkey = f"{authors[0].split()[-1].lower()}{year}{clean_title}" if authors else f"arxiv{year}"
    
    bibtex = f"@article{{{bibkey},\n"
    bibtex += f"  author    = {{{author_list}}},\n"
    bibtex += f"  title     = {{{title}}},\n"
    bibtex += f"  journal   = {{arXiv preprint arXiv:{arxiv_id}}},\n"
    bibtex += f"  year      = {{{year}}},\n"
    bibtex += f"  url       = {{{arxiv_url}}}\n"
    bibtex += "}"
    return bibtex

def generate_mock_figures(title, category):
    """
    Generates structured mock figure and table data which the React frontend can
    render as custom-drawn charts to simulate PDF figure extraction.
    Dynamically targets paper keywords to produce realistic, content-sensitive figures.
    """
    title_lower = title.lower()
    
    # 1. Reinforcement Learning / Robotics / Control
    if any(k in title_lower for k in ["policy", "reinforcement", "rl", "robot", "agent", "control", "manipulation", "action", "task", "game", "decision"]):
        nodes = [
            {"id": "Input", "label": "Environment State (S_t)", "x": 50, "y": 150, "type": "input"},
            {"id": "Encoder", "label": "Policy Network (Actor)", "x": 200, "y": 150, "type": "layer"},
            {"id": "Attention", "label": "Value Estimation (Critic)", "x": 350, "y": 100, "type": "layer"},
            {"id": "Linear", "label": "Action Dynamics Predictor", "x": 350, "y": 200, "type": "layer"},
            {"id": "Output", "label": "Action Output (A_t)", "x": 500, "y": 150, "type": "output"}
        ]
        links = [
            {"source": "Input", "target": "Encoder"},
            {"source": "Encoder", "target": "Attention"},
            {"source": "Encoder", "target": "Linear"},
            {"source": "Attention", "target": "Output"},
            {"source": "Linear", "target": "Output"}
        ]
        fig_caption = f"Figure 1: Actor-Critic policy mapping high-dimensional environment state observations S_t to control actions A_t in '{title[:40]}...'."
        
        xAxis = "Training Steps (x10^3)"
        yAxis = "Cumulative Episode Reward"
        series = [
            {"name": "Proposed Policy (Actor-Critic)", "points": [{"x": i, "y": float(np.round(150.0 * (1 - 0.96 ** (i/5)) + 10.0, 2))} for i in range(0, 101, 10)]},
            {"name": "PPO Baseline", "points": [{"x": i, "y": float(np.round(110.0 * (1 - 0.95 ** (i/5)) + 5.0, 2))} for i in range(0, 101, 10)]}
        ]
        chart_caption = "Figure 2: Average episode reward convergence comparison against policy gradient baselines over 100k training iterations."
        
        headers = ["Algorithm Setup", "Success Rate (%)", "Average Steps/Episode", "Inference Time (ms)"]
        rows = [
            ["Random Actions Baseline", "4.2", "450.0", "1.2"],
            ["Standard PPO Model", "72.8", "210.4", "18.3"],
            ["Proposed (Actor-Critic)", "91.5", "112.9", "22.0"],
            ["Proposed (Lightweight)", "88.0", "134.1", "11.5"]
        ]
        table_caption = f"Table 1: Benchmark comparison of the proposed policy performance on RL task domains."
        
    # 2. NLP / Transformers / Attention / LLMs
    elif any(k in title_lower for k in ["attention", "transformer", "llm", "language", "bert", "gpt", "semantic", "translation", "text", "prompt", "nlp", "speech", "token"]):
        nodes = [
            {"id": "Input", "label": "Input Token Embeddings", "x": 50, "y": 150, "type": "input"},
            {"id": "Encoder", "label": "Multi-Head Attention Layer", "x": 200, "y": 150, "type": "layer"},
            {"id": "Attention", "label": "Feed-Forward Projection", "x": 350, "y": 100, "type": "layer"},
            {"id": "Linear", "label": "Add & LayerNorm Residual", "x": 350, "y": 200, "type": "layer"},
            {"id": "Output", "label": "Softmax Token Probabilities", "x": 500, "y": 150, "type": "output"}
        ]
        links = [
            {"source": "Input", "target": "Encoder"},
            {"source": "Encoder", "target": "Attention"},
            {"source": "Encoder", "target": "Linear"},
            {"source": "Attention", "target": "Output"},
            {"source": "Linear", "target": "Output"}
        ]
        fig_caption = f"Figure 1: Transformer block pipeline including multi-head attention mapping and residual layers for '{title[:40]}...'."
        
        xAxis = "Fine-tuning Steps"
        yAxis = "Cross-Entropy Validation Loss"
        series = [
            {"name": "Proposed (Attention)", "points": [{"x": i, "y": float(np.round(2.5 * (0.94 ** (i/5)) + 0.35, 3))} for i in range(0, 101, 10)]},
            {"name": "BERT-Base Baseline", "points": [{"x": i, "y": float(np.round(2.5 * (0.94 ** (i/5)) + 0.55 + (0.01 * i if i > 80 else 0), 3))} for i in range(0, 101, 10)]}
        ]
        chart_caption = "Figure 2: Validation loss optimization trajectory during sequence-to-sequence fine-tuning checkpoints."
        
        headers = ["Model / Configurations", "GLUE Score (Avg)", "F1 score (SQuAD)", "Perplexity"]
        rows = [
            ["BERT-Base Baseline", "78.4", "0.772", "14.2"],
            ["BERT-Large (Baseline)", "81.2", "0.801", "11.3"],
            ["Proposed Architecture (Full)", "86.9", "0.865", "8.1"],
            ["Proposed (Pruned Heads)", "84.5", "0.842", "9.2"]
        ]
        table_caption = f"Table 1: NLP evaluation metrics comparison across popular linguistic understanding benchmarks."

    # 3. Computer Vision / Image Processing / Diffusion / CNN
    elif any(k in title_lower for k in ["image", "vision", "diffusion", "gan", "detection", "segmentation", "cnn", "resnet", "convolutional", "pixel", "depth", "classification", "video", "face"]):
        nodes = [
            {"id": "Input", "label": "Input Image Pixels", "x": 50, "y": 150, "type": "input"},
            {"id": "Encoder", "label": "Residual Convolutional Block", "x": 200, "y": 150, "type": "layer"},
            {"id": "Attention", "label": "Spatial Pooling / Attention", "x": 350, "y": 100, "type": "layer"},
            {"id": "Linear", "label": "Skip Connection Pathway", "x": 350, "y": 200, "type": "layer"},
            {"id": "Output", "label": "Segmentation / Classification Head", "x": 500, "y": 150, "type": "output"}
        ]
        links = [
            {"source": "Input", "target": "Encoder"},
            {"source": "Encoder", "target": "Attention"},
            {"source": "Encoder", "target": "Linear"},
            {"source": "Attention", "target": "Output"},
            {"source": "Linear", "target": "Output"}
        ]
        fig_caption = f"Figure 1: Spatial pooling networks and convolutional layers with residual connections for '{title[:40]}...'."
        
        xAxis = "Training Epochs"
        yAxis = "Top-1 Validation Accuracy (%)"
        series = [
            {"name": "Proposed ResNet Variant", "points": [{"x": i, "y": float(np.round(89.0 * (1 - 0.95 ** (i/8)) + 5.0, 2))} for i in range(0, 101, 10)]},
            {"name": "ResNet-50 Baseline", "points": [{"x": i, "y": float(np.round(82.0 * (1 - 0.94 ** (i/8)) + 4.0, 2))} for i in range(0, 101, 10)]}
        ]
        chart_caption = "Figure 2: Top-1 validation accuracy progression on ImageNet dataset benchmarks across 100 training epochs."
        
        headers = ["Backbone Network", "mAP@0.5 (%)", "Inference Latency (ms)", "Parameters (M)"]
        rows = [
            ["ResNet-50 Backbone", "64.1", "24.5", "25.6"],
            ["ViT-Base Backbone", "71.3", "45.0", "86.0"],
            ["Proposed Framework (Ours)", "78.9", "18.2", "29.4"],
            ["Proposed Lite Model", "74.8", "11.0", "14.2"]
        ]
        table_caption = f"Table 1: Ablation comparison of spatial segmentation and mAP performance using various backbone networks."

    # 4. Default: General Machine Learning / Optimizations
    else:
        nodes = [
            {"id": "Input", "label": "High-dim Vector Input", "x": 50, "y": 150, "type": "input"},
            {"id": "Encoder", "label": "Latent Feature Projection", "x": 200, "y": 150, "type": "layer"},
            {"id": "Attention", "label": "Batch Normalization / Dropout", "x": 350, "y": 100, "type": "layer"},
            {"id": "Linear", "label": "Dense Linear Layers", "x": 350, "y": 200, "type": "layer"},
            {"id": "Output", "label": "Relevance Classifier Output", "x": 500, "y": 150, "type": "output"}
        ]
        links = [
            {"source": "Input", "target": "Encoder"},
            {"source": "Encoder", "target": "Attention"},
            {"source": "Encoder", "target": "Linear"},
            {"source": "Attention", "target": "Output"},
            {"source": "Linear", "target": "Output"}
        ]
        fig_caption = f"Figure 1: Deep learning feed-forward pipeline projection for multi-scale predictive relevance mapping in '{title[:40]}...'."
        
        xAxis = "Optimization Epochs"
        yAxis = "Validation MSE Error"
        series = [
            {"name": "Proposed (Optimized)", "points": [{"x": i, "y": float(np.round(1.2 * (0.95 ** (i/5)) + 0.05, 3))} for i in range(0, 101, 10)]},
            {"name": "Standard SGD Optimizer", "points": [{"x": i, "y": float(np.round(1.2 * (0.96 ** (i/5)) + 0.12, 3))} for i in range(0, 101, 10)]}
        ]
        chart_caption = "Figure 2: Mean Squared Error convergence rate comparison under Adam vs SGD optimization schedules."
        
        headers = ["Model Settings", "Accuracy (%)", "F1 Score", "Parameter count"]
        rows = [
            ["Linear Baseline Model", "72.4", "0.710", "12K"],
            ["MLP MLP (Baseline)", "81.2", "0.801", "4.2M"],
            ["Proposed Framework (Ours)", "86.9", "0.865", "12.8M"],
            ["Proposed Lightweight Model", "84.5", "0.842", "6.1M"]
        ]
        table_caption = f"Table 1: Statistical evaluation metrics comparison on classification test splits."

    return [
        {
            "id": "fig_1",
            "type": "figure",
            "title": "Proposed Model Architecture Pipeline",
            "caption": fig_caption,
            "data": {
                "nodes": nodes,
                "links": links
            }
        },
        {
            "id": "fig_2",
            "type": "chart",
            "title": "Model Performance & Optimization Curves",
            "caption": chart_caption,
            "data": {
                "xAxis": xAxis,
                "yAxis": yAxis,
                "series": series
            }
        },
        {
            "id": "tab_1",
            "type": "table",
            "title": "Quantitative Ablation & Benchmark Comparisons",
            "caption": table_caption,
            "data": {
                "headers": headers,
                "rows": rows
            }
        }
    ]

def fetch_arxiv_papers(limit=200):
    """
    Queries arXiv API for recent machine learning and computer science papers.
    """
    print(f"Fetching {limit} papers from arXiv...")
    query = "cat:cs.LG+OR+cat:cs.CV+OR+cat:cs.CL+OR+cat:cs.AI"
    url = f"http://export.arxiv.org/api/query?search_query={query}&max_results={limit}&sortBy=submittedDate&sortOrder=descending"
    
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
    except Exception as e:
        print(f"Error fetching data from arXiv: {e}")
        return []
        
    root = ET.fromstring(xml_data)
    
    # XML namespace mapping
    namespaces = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }
    
    papers = []
    for entry in root.findall("atom:entry", namespaces):
        title = clean_text(entry.find("atom:title", namespaces).text)
        abstract = clean_text(entry.find("atom:summary", namespaces).text)
        
        arxiv_url = entry.find("atom:id", namespaces).text.strip()
        published = entry.find("atom:published", namespaces).text.strip()
        
        authors = [clean_text(author.find("atom:name", namespaces).text) for author in entry.findall("atom:author", namespaces)]
        
        # Primary category
        primary_cat_el = entry.find("arxiv:primary_category", namespaces)
        if primary_cat_el is not None:
            primary_cat = primary_cat_el.attrib.get("term")
        else:
            category_els = entry.findall("atom:category", namespaces)
            primary_cat = category_els[0].attrib.get("term") if category_els else "cs.LG"
            
        category_name = ARXIV_CATEGORIES.get(primary_cat, "Machine Learning")
        
        # Quality Filters: Prune incomplete, stub, or out-of-domain publications
        if len(abstract) < 350:
            continue
        if len(title) < 15:
            continue
        if not authors:
            continue
        if primary_cat not in ARXIV_CATEGORIES:
            continue
            
        # Academic indicators test: ensure abstract contains robust scientific vocabulary
        quality_indicators = ["proposed", "method", "results", "model", "performance", "framework", "dataset", "accuracy", "learning", "network", "benchmark"]
        abstract_lower = abstract.lower()
        indicator_matches = sum(1 for indicator in quality_indicators if indicator in abstract_lower)
        if indicator_matches < 3:
            continue
            
        highlight = extract_key_sentence(abstract)
        bibtex = generate_bibtex(arxiv_url, title, authors, published)
        figures = generate_mock_figures(title, category_name)
        
        papers.append({
            "id": arxiv_url.split("/abs/")[-1].split("v")[0], # Unique ID based on arXiv ID
            "arxiv_url": arxiv_url,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "published": published[:10], # YYYY-MM-DD
            "primary_category": primary_cat,
            "category_name": category_name,
            "highlight": highlight,
            "bibtex": bibtex,
            "figures": figures,
            "relevance_score": 0.0 # Default starting relevance score
        })
        
    return papers

def filter_papers_with_gemini(papers, api_key):
    """
    Uses Gemini API to screen papers and reject low-quality, draft, or out-of-domain publications.
    Processes papers in batches of 25 to optimize latency and api quota.
    """
    if not api_key:
        return papers
        
    print(f"Agent Quality Filter: Screening {len(papers)} papers using Gemini...")
    rejected_ids = set()
    
    # To avoid excessive api latency and quota usage, we only filter the top 150 papers
    # (which are the most recent ones shown on the maps and digest)
    candidate_limit = 150
    candidates = papers[:candidate_limit]
    remaining = papers[candidate_limit:]
    
    batch_size = 25
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        batch_data = [{"id": p["id"], "title": p["title"], "abstract": p["abstract"][:300]} for p in batch]
        
        prompt = f"""
You are a scientific program committee chair. Evaluate the quality of these papers based on their titles and abstracts.
Identify any paper that is a stub, draft, incomplete, has low academic rigor, or is out of domain (not AI/ML/CV/NLP).

Papers list:
{json.dumps(batch_data)}

Return a strict JSON list of IDs of the papers that are LOW QUALITY, INCOMPLETE, or OUT-OF-DOMAIN and should be rejected.
Format your response as a valid JSON list. Do not use markdown blocks or HTML.
Example: ["id1", "id2"]
"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        headers = {"Content-Type": "application/json"}
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                res_text = res_data['candidates'][0]['content']['parts'][0]['text']
                rejected = json.loads(res_text.strip())
                if isinstance(rejected, list):
                    for rid in rejected:
                        rejected_ids.add(str(rid))
        except Exception as e:
            print(f"Gemini filter batch {i//batch_size} failed: {e}. Skipping batch filtering.")
            
    # Filter out rejected papers
    filtered_candidates = [p for p in candidates if p["id"] not in rejected_ids]
    print(f"Agent Quality Filter: Screened candidates. Rejected {len(candidates) - len(filtered_candidates)} papers.")
    
    return filtered_candidates + remaining

def load_or_process_papers():
    """
    Loads papers from JSON database cache. If not found, crawls arXiv, computes
    dimensionality reduction (t-SNE) on paper embeddings, and saves to file.
    """
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                papers = json.load(f)
                # Dynamic re-generation of figures based on content keywords
                for p in papers:
                    p["figures"] = generate_mock_figures(p["title"], p["category_name"])
                return papers
        except Exception as e:
            print(f"Error loading cached papers: {e}. Crawling again...")

    # Fetch fresh papers and filter for quality, keeping up to 1500 top papers
    papers = fetch_arxiv_papers(3000)[:1500]
    
    # To protect your tokens, AI-powered screening is DISABLED by default.
    # It will only execute if the environment flag 'ENABLE_CRAWLER_AI' is explicitly set to 'true'.
    enable_crawler_ai = os.environ.get("ENABLE_CRAWLER_AI", "false").lower() == "true"
    api_key = os.environ.get("GEMINI_API_KEY")
    if enable_crawler_ai and api_key:
        try:
            papers = filter_papers_with_gemini(papers, api_key)
        except Exception as e:
            print(f"AI Filter Agent failed: {e}. Falling back to default list.")
            
    if not papers:
        # Fallback to local mockup list if connection fails
        print("Fallback to local mockup papers...")
        papers = get_fallback_papers()
        
    # Precompute TF-IDF embeddings and t-SNE
    corpus = [f"{p['title']} [SEP] {p['abstract']}" for p in papers]
    
    vectorizer = TfidfVectorizer(max_features=256, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus).toarray()
    
    # Run t-SNE for 2D Map Projection
    n_samples = len(papers)
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
    
    print(f"Running t-SNE dimensionality reduction (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", n_iter=1000)
    coords_2d = tsne.fit_transform(tfidf_matrix)
    
    # Normalize coordinates to [-100, 100] for visual UI scaling
    min_x, max_x = coords_2d[:, 0].min(), coords_2d[:, 0].max()
    min_y, max_y = coords_2d[:, 1].min(), coords_2d[:, 1].max()
    
    scale_x = (max_x - min_x) if (max_x - min_x) != 0 else 1
    scale_y = (max_y - min_y) if (max_y - min_y) != 0 else 1
    
    coords_normalized = [
        [
            float(((x - min_x) / scale_x) * 180 - 90), # margin offset
            float(((y - min_y) / scale_y) * 180 - 90)
        ]
        for x, y in coords_2d
    ]
    
    # Attach coordinates and embeddings (truncated or stored as floats) to papers list
    for idx, paper in enumerate(papers):
        paper["x"] = coords_normalized[idx][0]
        paper["y"] = coords_normalized[idx][1]
        paper["embedding"] = tfidf_matrix[idx].tolist() # Stored for logistic regression
        
    # Write to local cache
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(papers, f, indent=2)
        
    return papers

def get_fallback_papers():
    """
    Generates a list of 5 fallback papers in case arXiv is down.
    """
    mock_list = []
    topics = [
        ("Attention Is All You Need", "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.", "cs.CL"),
        ("Deep Residual Learning for Image Recognition", "We present a residual learning framework to ease the training of networks that are substantially deeper.", "cs.CV"),
        ("Generative Adversarial Nets", "We propose a new framework for estimating generative models via an adversarial process.", "cs.LG"),
        ("BERT: Pre-training of Deep Bidirectional Transformers", "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations.", "cs.CL"),
        ("YOLOv4: Optimal Speed and Accuracy of Object Detection", "We analyze a large number of features and select an optimal combination to improve object detection speed.", "cs.CV")
    ]
    
    for idx, (title, abstract, category) in enumerate(topics):
        arxiv_url = f"http://arxiv.org/abs/2101.{idx:04d}v1"
        cat_name = ARXIV_CATEGORIES.get(category, "Machine Learning")
        mock_list.append({
            "id": f"2101.{idx:04d}",
            "arxiv_url": arxiv_url,
            "title": title,
            "abstract": abstract,
            "authors": [f"Author {chr(65+idx)}", "Collaborator B"],
            "published": "2025-01-01",
            "primary_category": category,
            "category_name": cat_name,
            "highlight": abstract,
            "bibtex": f"@article{{mock{idx},\n  title={{{title}}}\n}}",
            "figures": generate_mock_figures(title, cat_name),
            "relevance_score": 0.0
        })
    return mock_list

if __name__ == "__main__":
    load_or_process_papers()
    print("Papers processing completed!")
