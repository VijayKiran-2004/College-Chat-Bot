import asyncio
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock
# Add project root to path (one level up from tests/)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import backend
from app.services.logger_service import ResponseLogger
from backend import chat, QueryRequest

async def test_logging():
    print("Testing logging integration...")
    
    # Setup
    log_dir = Path("test_logs")
    if log_dir.exists():
        shutil.rmtree(log_dir)
    
    # Mock Router
    mock_router = MagicMock()
    mock_router.route_query.return_value = "This is a mock response."
    backend.query_router = mock_router
    
    # Initialize Logger
    backend.response_logger = ResponseLogger(log_dir=str(log_dir), filename="test_log.csv")
    
    # Send Query
    request = QueryRequest(message="Hello test", session_id="test_session")
    response = await chat(request)
    
    # Verify
    log_file = log_dir / "test_log.csv"
    if not log_file.exists():
        print("FAIL: Log file not created.")
        return
    
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
        print(f"Log Content:\n{content}")
        
        if "Hello test" in content and "This is a mock response" in content:
            print("SUCCESS: Log contains query and response.")
        else:
            print("FAIL: Log missing content.")

    # Cleanup
    if log_dir.exists():
        shutil.rmtree(log_dir)

if __name__ == "__main__":
    asyncio.run(test_logging())
