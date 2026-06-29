import sys
import os

# Adjust path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import load_or_process_papers
from core.recommender import train_recommender

def run_tests():
    print("=== Testing Data Loading and Processing ===")
    papers = load_or_process_papers()
    print(f"Successfully loaded/processed {len(papers)} papers.")
    assert len(papers) > 0, "No papers were loaded!"
    
    # Assert coordinates and embedding are present
    sample_paper = papers[0]
    print(f"Sample paper: '{sample_paper['title']}'")
    print(f"Coordinates: x={sample_paper['x']:.2f}, y={sample_paper['y']:.2f}")
    assert "x" in sample_paper and "y" in sample_paper, "Coordinates 'x' and 'y' missing!"
    assert "embedding" in sample_paper, "Embedding missing!"
    assert len(sample_paper["embedding"]) == 256, f"Expected 256 dim embedding, got {len(sample_paper['embedding'])}"
    
    print("\n=== Testing Recommender Model Training ===")
    # Simulate some ratings:
    # 2 positive upvotes, 1 negative downvote
    mock_ratings = {
        papers[0]["id"]: 1,
        papers[1]["id"]: 1,
        papers[2]["id"]: -1
    }
    
    print("Training with mock ratings...")
    updated_papers = train_recommender(papers, mock_ratings)
    
    # Check updated relevance scores
    scores = [p["relevance_score"] for p in updated_papers]
    print(f"Relevance scores range: Min={min(scores):.1f}, Max={max(scores):.1f}")
    
    # Check that upvoted papers have positive, relatively high scores
    upvoted_scores = [p["relevance_score"] for p in updated_papers if p["id"] in mock_ratings and mock_ratings[p["id"]] == 1]
    downvoted_scores = [p["relevance_score"] for p in updated_papers if p["id"] in mock_ratings and mock_ratings[p["id"]] == -1]
    
    print(f"Upvoted paper scores: {upvoted_scores}")
    print(f"Downvoted paper scores: {downvoted_scores}")
    
    assert all(s >= -100.0 and s <= 100.0 for s in scores), "Relevance scores out of [-100, 100] range!"
    assert all(us > ds for us in upvoted_scores for ds in downvoted_scores), "Upvoted papers should have higher relevance than downvoted ones!"
    
    print("\nRecommender logic tests PASSED successfully!")

if __name__ == "__main__":
    run_tests()
