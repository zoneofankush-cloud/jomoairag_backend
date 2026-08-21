from langchain_community.utilities import SearxSearchWrapper
import json

def test_searxng_search(query: str):
    """
    Test function to verify the SearxNG connection and JSON output.
    """
    print(f"Testing SearxNG Search for query: '{query}'...")
    
    try:
        searx = SearxSearchWrapper(searx_host="http://localhost:8080")
        
        # .results() returns structured dicts which is perfect for AI agents
        results = searx.results(query, num_results=5)
        
        print("\n--- Search Results ---")
        for idx, res in enumerate(results, 1):
            print(f"\nResult {idx}:")
            print(f"Title: {res.get('title')}")
            print(f"Link: {res.get('link')}")
            print(f"Snippet: {res.get('snippet')}")
            
        return results
        
    except Exception as e:
        print(f"\n[ERROR] SearxNG Connection failed!")
        print(f"Details: {e}")
        print("\nMake sure your SearxNG docker container is running and JSON format is enabled in settings.yml!")

if __name__ == "__main__":
    test_searxng_search("Latest advances in artificial intelligence 2026")
