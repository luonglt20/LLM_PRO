import random
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_recommender(papers, ratings, C=0.1, V=0.8, S=5.0):
    """
    Trains a Logistic Regression recommender model based on user ratings
    using the custom weighting scheme described in Section 3.1.1.
    
    ratings: dict of {paper_id: int} where 1 = upvote, -1 = downvote
    """
    # 1. Identify groups
    positive_papers = []
    negative_papers = []
    unrated_papers = []
    
    for paper in papers:
        pid = paper["id"]
        if pid in ratings:
            if ratings[pid] == 1:
                positive_papers.append(paper)
            elif ratings[pid] == -1:
                negative_papers.append(paper)
        else:
            unrated_papers.append(paper)
            
    n_P = len(positive_papers)
    n_N = len(negative_papers)
    
    # If there are no positive ratings, we cannot train a classifier.
    # Return 0.0 relevance scores for all papers.
    if n_P == 0:
        for paper in papers:
            paper["relevance_score"] = 0.0
        return papers
        
    # 2. Sample random negatives (nR)
    # The paper uses 5k random negatives, but for our demo we sample up to 100
    # to keep training instant on local systems.
    n_R_target = min(100, len(unrated_papers))
    random_negatives = random.sample(unrated_papers, n_R_target) if n_R_target > 0 else []
    n_R = len(random_negatives)
    
    # 3. Assemble training set
    X_train = []
    y_train = []
    sample_weights = []
    
    n_T = n_P + n_N + n_R
    
    # Calculate intermediate weights
    # w_tilde_P = 1 / n_P
    # w_tilde_N = S * V / (V * n_N + (1 - V) * n_R)
    # w_tilde_R = S * (1 - V) / (V * n_N + (1 - V) * n_R)
    
    denom = V * n_N + (1.0 - V) * n_R
    if denom == 0:
        denom = 1.0 # Prevent division by zero if both are 0
        
    # Scaled weights: w = n_T * w_tilde
    w_P = n_T / n_P
    w_N = n_T * (S * V) / denom
    w_R = n_T * (S * (1.0 - V)) / denom
    
    # Add positives
    for paper in positive_papers:
        X_train.append(paper["embedding"])
        y_train.append(1)
        sample_weights.append(w_P)
        
    # Add explicit negatives
    for paper in negative_papers:
        X_train.append(paper["embedding"])
        y_train.append(0)
        sample_weights.append(w_N)
        
    # Add random negatives
    for paper in random_negatives:
        X_train.append(paper["embedding"])
        y_train.append(0)
        sample_weights.append(w_R)
        
    # 4. Train Logistic Regression model
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    sample_weights = np.array(sample_weights)
    
    # If we only have 1 class (e.g. n_N = 0 and n_R = 0, which is extremely rare),
    # we can't fit a logistic regression.
    if len(np.unique(y_train)) < 2:
        for paper in papers:
            paper["relevance_score"] = 0.0
        return papers
        
    # Fit model with L2 regularization (C is inverse regularization parameter)
    clf = LogisticRegression(C=C, penalty="l2", solver="liblinear", random_state=42)
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    
    # 5. Predict relevance scores for all papers
    all_embeddings = np.array([p["embedding"] for p in papers])
    probs = clf.predict_proba(all_embeddings)[:, 1] # Probability of class 1 (positive)
    
    # The paper: "We linearly scale the output of our model to [-100, 100] and display this relevance value"
    # Probability in [0, 1] maps to [-100, 100]
    scores = (probs - 0.5) * 200.0
    
    for idx, paper in enumerate(papers):
        paper["relevance_score"] = float(np.round(scores[idx], 1))
        
    return papers
