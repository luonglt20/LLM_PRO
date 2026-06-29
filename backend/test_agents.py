import os
import sys

# Ensure backend directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_orchestrator import run_agent_workflow

def test_multi_agent_workflow():
    print("Testing Multi-Agent LangGraph Workflow...")
    query = "Sequential dexterity and pretrained policies"
    
    # Run workflow without api key (uses local heuristic fallback)
    output = run_agent_workflow(query, api_key=None)
    
    # Assert state keys exist
    assert "final_report" in output, "Workflow output missing final_report"
    assert "agent_logs" in output, "Workflow output missing agent_logs"
    assert "papers" in output, "Workflow output missing papers list"
    
    report = output["final_report"]
    logs = output["agent_logs"]
    papers = output["papers"]
    
    print(f"\nExtracted Papers count: {len(papers)}")
    print(f"Total Logs captured: {len(logs)}")
    
    # Assert logs structure
    for idx, log in enumerate(logs):
        print(f"Log {idx+1}: [{log['agent']}] -> {log['message']}")
        assert "agent" in log, "Log missing agent identifier"
        assert "message" in log, "Log missing text message"
        
    # Assert that all 3 agents executed
    executed_agents = {log["agent"] for log in logs}
    expected_agents = {"ArxivSearchAgent", "PaperCriticAgent", "LiteratureReviewAgent"}
    
    print(f"\nExecuted Agents: {executed_agents}")
    assert expected_agents.issubset(executed_agents), f"Missing some expected agents. Got: {executed_agents}"
    
    # Assert report structure
    print(f"\nFinal Compiled Report:\n{report[:300]}...")
    assert len(report) > 100, "Report body too short"
    assert "# Literature Review" in report, "Report missing main header"
    assert "Introduction" in report, "Report missing Introduction section"
    assert "Methodology" in report, "Report missing Methodology section"
    
    print("\n[SUCCESS] Multi-Agent LangGraph integration test completed successfully!")

if __name__ == "__main__":
    test_multi_agent_workflow()
