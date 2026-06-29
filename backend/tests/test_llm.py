import sys
import os

# Adjust path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_helper import explain_recommendation

def run_tests():
    print("=== Testing LLM Helper ===")
    
    # Mock data
    user_interests = [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers"
    ]
    
    paper_title = "DexCompose: Reusing Dexterous Policies for Multi-Task Manipulation"
    paper_abstract = "We propose Compose, a framework based on self-attention for learning policies. Our method leverages transformer blocks to compose actions."
    category_name = "Machine Learning"
    
    # Test 1: Fallback (unsetting GEMINI_API_KEY environment variable if exists)
    original_key = os.environ.get("GEMINI_API_KEY")
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
        
    print("Test 1: Running without API Key (Local Heuristic Fallback)...")
    res_fallback = explain_recommendation(user_interests, paper_title, paper_abstract, category_name)
    
    print(f"Source: {res_fallback.get('source')}")
    print(f"Explanation: {res_fallback.get('explanation')}")
    print(f"Summary: {res_fallback.get('tailored_summary')}")
    
    assert res_fallback["source"] == "Local Heuristic", "Expected fallback to trigger!"
    assert "explanation" in res_fallback and "tailored_summary" in res_fallback, "Missing keys in result!"
    assert "attention" in res_fallback["explanation"].lower() or "attention" in res_fallback["tailored_summary"].lower(), "Heuristic should match the word 'attention'!"
    
    # Test 2: Graceful error recovery with invalid key
    os.environ["GEMINI_API_KEY"] = "INVALID_API_KEY_TEST"
    print("\nTest 2: Running with Invalid API Key (Should fallback gracefully)...")
    res_invalid_key = explain_recommendation(user_interests, paper_title, paper_abstract, category_name)
    
    print(f"Source: {res_invalid_key.get('source')}")
    assert res_invalid_key["source"] == "Local Heuristic", "Expected API failure to fall back gracefully!"
    
    # Restore key if was present
    if original_key:
        os.environ["GEMINI_API_KEY"] = original_key
        
    print("\nLLM Helper tests PASSED successfully!")

if __name__ == "__main__":
    run_tests()
