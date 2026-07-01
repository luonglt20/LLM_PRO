import os
import json
import urllib.request
import urllib.error
import re

def get_keywords(text):
    """
    Extracts meaningful keywords from text by filtering stop words.
    """
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'of', 'in', 'on', 'at', 
        'to', 'for', 'with', 'by', 'about', 'against', 'between', 'into', 'through', 'during', 
        'before', 'after', 'above', 'below', 'from', 'up', 'down', 'in', 'out', 'off', 'over', 
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
        'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
        'can', 'will', 'just', 'don', 'should', 'now', 'using', 'reusing', 'optimal', 'speed', 
        'accuracy', 'single', 'multi', 'composed', 'task'
    }
    # Clean and split
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return {w for w in words if w not in stop_words}

def explain_recommendation_heuristic(user_interests, paper_title, paper_abstract, category_name):
    """
    Local NLP heuristic that performs keyword overlapping to explain
    relevance and generate a summary when Gemini API is offline.
    """
    # Combine upvoted paper titles for interest keywords
    interest_text = " ".join(user_interests)
    interest_keywords = get_keywords(interest_text)
    
    # Target paper keywords
    paper_text = f"{paper_title} {paper_abstract}"
    paper_keywords = get_keywords(paper_text)
    
    # Calculate overlap
    overlap = interest_keywords.intersection(paper_keywords)
    matched_words = list(overlap)[:3] # Grab top 3 overlaps
    
    # Generate explanation
    if matched_words:
        keywords_str = ", ".join(f"'{w}'" for w in matched_words)
        explanation = f"Recommended because you liked papers matching keyword(s) {keywords_str}, which are core to this {category_name} research."
        tailored_summary = f"This work advances techniques in {category_name} focusing on {matched_words[0]} mechanisms. It aligns with your interest in {keywords_str} optimizations."
    else:
        explanation = f"Recommended based on your interest in {category_name} models, sharing general concepts with your bookmarked publications."
        tailored_summary = f"This paper presents new methodologies within {category_name}. It proposes optimizations that run parallel to your reading habits."
        
    return {
        "explanation": explanation,
        "tailored_summary": tailored_summary,
        "source": "Local Heuristic"
      }

def explain_recommendation_llm(user_interests, paper_title, paper_abstract, category_name, api_key):
    """
    Calls Google Gemini or Groq Llama API to get structured JSON recommendations explanation.
    """
    # Build prompt
    prompt = f"""
You are an expert scientific paper recommendation assistant for researchers.
The user has expressed strong interest in these papers they upvoted:
{chr(10).join(f"- {title}" for title in user_interests)}

Now, the user is looking at this recommended paper:
Title: {paper_title}
Category: {category_name}
Abstract: {paper_abstract}

Your tasks:
1. "explanation": Write a 1-sentence personalized explanation (in Vietnamese) detailing why this paper matches their upvoted interest profile. Be specific about methodologies or concepts. Do not use any markdown formatting characters (like *, **, #) in the explanation string.
2. "tailored_summary": Write a 2-sentence summary (in Vietnamese) of this paper tailored to their background, highlighting details they would find most interesting based on their upvotes. Do not use any markdown formatting characters (like *, **, #) in the summary string.

You must respond in strict JSON format matching the schema. Do not write markdown blocks or HTML.
"""
    
    if api_key.startswith("gsk_"):
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        try:
            req = urllib.request.Request(
                url, 
                data=json.dumps(payload).encode('utf-8'), 
                headers=headers, 
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                text_response = res_data['choices'][0]['message']['content']
                parsed_json = json.loads(text_response.strip())
                parsed_json["source"] = "Groq Llama"
                return parsed_json
        except Exception as e:
            print(f"Error calling Groq API in explain_recommendation_llm: {e}. Falling back to heuristic...")
            return explain_recommendation_heuristic(user_interests, paper_title, paper_abstract, category_name)
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "explanation": {"type": "STRING"},
                        "tailored_summary": {"type": "STRING"}
                    },
                    "required": ["explanation", "tailored_summary"]
                }
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
            with urllib.request.urlopen(req, timeout=8) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                text_response = res_data['candidates'][0]['content']['parts'][0]['text']
                parsed_json = json.loads(text_response.strip())
                parsed_json["source"] = "Gemini LLM"
                return parsed_json
        except Exception as e:
            print(f"Error calling Gemini API: {e}. Falling back to heuristic...")
            return explain_recommendation_heuristic(user_interests, paper_title, paper_abstract, category_name)

def explain_recommendation(user_interests, paper_title, paper_abstract, category_name, api_key=None):
    """
    Main entrypoint for recommendations explanations. 
    Checks for passed api_key or GEMINI_API_KEY environment variable.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
        
    if not api_key:
        # Fallback instantly if no API key is provided
        return explain_recommendation_heuristic(user_interests, paper_title, paper_abstract, category_name)
        
    return explain_recommendation_llm(user_interests, paper_title, paper_abstract, category_name, api_key)

def get_gemini_embedding(text, api_key):
    """
    Fetches 768-dimensional dense vector embeddings using Google's text-embedding-004 model.
    Returns None on failure so the application falls back to local TF-IDF similarity.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
    payload = {
        "model": "models/text-embedding-004",
        "content": {
            "parts": [{"text": text}]
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
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            return res_data['embedding']['values']
    except Exception as e:
        print(f"Error fetching Gemini embedding: {e}. Falling back...")
        return None

def answer_pdf_question_heuristic(chunks, question):
    """
    Local heuristic fallback that displays retrieved snippets from the PDF
    when Gemini API key is offline or unconfigured.
    """
    ans = "🤖 [Chế độ dự phòng Local]\n\n"
    ans += "Gemini API Key chưa được cấu hình hoặc đã hết hạn/hết hạn ngạch. Dưới đây là các phần nội dung liên quan nhất được tìm thấy từ tài liệu PDF:\n\n"
    for i, c in enumerate(chunks[:2]):
        cleaned_text = c['text'].replace('\n', ' ').strip()
        score = c.get('score', 0.0)
        ans += f"📄 Phần {i+1} (Điểm tương đồng: {score:.2f}):\n"
        ans += f"\"... {cleaned_text[:350]} ...\"\n\n"
    ans += "💡 Mẹo: Hãy nhập Gemini API Key chính xác trong menu cài đặt (biểu tượng 🔑 ở góc trên bên phải) để mở khóa câu trả lời AI đầy đủ và thông minh nhất!"
    return ans

def answer_pdf_question(chunks, question, api_key):
    """
    Asks Gemini or Groq to answer a user's question using the retrieved PDF chunks as context.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
        
    if not api_key:
        return answer_pdf_question_heuristic(chunks, question)
        
    snippets = ""
    for i, c in enumerate(chunks):
        snippets += f"\nSnippet {i+1} (Page {c['page_idx']}):\n{c['text']}\n"
        
    prompt = f"""
You are an expert scientific research assistant.
Answer the user's question about the paper based ONLY on the provided context snippets extracted from the paper's PDF.
Keep your response concise, factual, and professional. Do not use any markdown formatting characters (like *, **, #, etc.) in your response. Write only in clean plain text with regular paragraphs or clean bullet points (using standard hyphen - without stars).
If the context snippets do not contain enough information to answer the question, state that clearly.

Context Snippets:
{snippets}

Question: {question}
Answer (in Vietnamese):
"""
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
            "Authorization": f"Bearer {api_key}"
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                return res_data['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error calling Groq in answer_pdf_question: {e}. Falling back...")
            return answer_pdf_question_heuristic(chunks, question)
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
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
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                return res_data['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            print(f"Error calling Gemini in answer_pdf_question: {e}. Falling back...")
            return answer_pdf_question_heuristic(chunks, question)

