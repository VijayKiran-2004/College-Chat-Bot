import sys
import os

# Add parent directory to path to allow importing app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.logger_service import ResponseLogger

try:
    logger = ResponseLogger(log_dir="logs")
    print(f"Logging to: {logger.log_file}")
    
    logger.log_response(
        user_query="Test Query for Excel",
        bot_response="This is a test response to verify Excel logging.",
        time_taken=0.1234,
        session_id="TEST-SESSION-123"
    )
    print("Log entry written successfully.")

except Exception as e:
    print(f"Test failed: {e}")
