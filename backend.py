#!/usr/bin/env python3
"""
FastAPI Backend Server for College Chatbot
Provides REST API endpoints for the frontend to interact with UltraRAG
"""

import sys
from pathlib import Path
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time
from contextlib import asynccontextmanager
from fastapi import Request

# Add app directory to path
# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.query_router import QueryRouter
from app.services.logger_service import ResponseLogger

# Initialize global logger
response_logger = None


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class QueryRequest(BaseModel):
    """Request model for chat queries"""
    message: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for chat queries"""
    answer: str
    session_id: str


# ============================================================
# FASTAPI APP SETUP
# ============================================================

# ============================================================
# LIFESPAN & STARTUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    Measures and logs backend setup time.
    """
    print("\n" + "=" * 70)
    print("COLLEGE BUDDY - STARTING UP")
    start_time = time.time()
    
    # Initialize Systems
    initialize_systems()
    
    elapsed = time.time() - start_time
    print(f"✓ Backend initialized in {elapsed:.4f} seconds")
    print("=" * 70 + "\n")
    
    yield
    
    # Shutdown logic (if any)
    print("Shutting down...")


app = FastAPI(
    title="College Buddy Backend",
    description="REST API for College Chatbot using UltraRAG",
    version="1.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to measure request processing time
    Adds 'X-Process-Time' header to response
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add header
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log slow requests
    if process_time > 1.0:
        print(f"⚠ Slow Request: {request.url.path} took {process_time:.4f}s")
    else:
        print(f"ℹ Request: {request.url.path} took {process_time:.4f}s")
        
    return response

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GLOBAL STATE
# ============================================================

# Initialize Router on startup
query_router = None
session_history = {}  # Store chat history by session_id


def initialize_systems():
    """Initialize Query Router and subsystems"""
    global query_router, response_logger
    print("\n" + "=" * 70)
    print("COLLEGE BUDDY - BACKEND SERVER")
    print("=" * 70)
    print("\nInitializing Systems...")
    try:
        response_logger = ResponseLogger()
        print("✓ Response Logger initialized")
        query_router = QueryRouter()
        print("✓ Query Router & Unified Brain initialized successfully!")
        return True
    except Exception as e:
        print(f"✗ Error initializing systems: {e}")
        print("Make sure:")
        print("  1. All data files are present")
        print("  2. Ollama is running: ollama serve")
        print("  3. Model is pulled: ollama pull gemma2:2b")
        return False


 


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """Root endpoint - server health check"""
    return {
        "status": "online",
        "service": "College Buddy Backend",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /query",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if query_router is None:
        raise HTTPException(status_code=503, detail="Systems not initialized")
    
    return {
        "status": "healthy",
        "router": "ready",
        "sessions_active": len(session_history)
    }


@app.post("/query", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """
    Main chat endpoint - accepts user queries and returns responses
    
    Request:
        - message: User's question
        - session_id: Optional session ID for conversation continuity
    
    Response:
        - answer: Chatbot's response
        - session_id: Session ID for this conversation
    """
    
    if query_router is None:
        raise HTTPException(
            status_code=503, 
            detail="System not initialized. Please check the server logs and ensure all dependencies are installed."
        )
    
    # Validate message
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Create or use existing session
        if request.session_id is None:
            session_id = str(uuid.uuid4())
            session_history[session_id] = []
        else:
            session_id = request.session_id
            if session_id not in session_history:
                session_history[session_id] = []
        
        # Store user message in history
        session_history[session_id].append({
            "role": "user",
            "message": request.message
        })
        
        # Generate response using Unified Query Router
        start_time = time.time()
        answer = query_router.route_query(request.message)
        end_time = time.time()
        response_time = end_time - start_time
        
        # Log the response
        try:
            if response_logger:
                response_logger.log_response(
                    user_query=request.message,
                    bot_response=str(answer),
                    time_taken=response_time,
                    session_id=session_id
                )
        except Exception as log_err:
            print(f"Logging failed: {log_err}")

        # Store bot response in history
        session_history[session_id].append({
            "role": "bot",
            "message": answer
        })
        
        # Keep session history manageable (last 50 exchanges)
        if len(session_history[session_id]) > 100:
            session_history[session_id] = session_history[session_id][-100:]
        
        return QueryResponse(
            answer=str(answer),
            session_id=session_id
        )
    
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/session/{session_id}")
async def get_session_history(session_id: str):
    """
    Retrieve chat history for a session
    
    Args:
        session_id: The session ID
    
    Returns:
        Chat history for the session
    """
    if session_id not in session_history:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "history": session_history[session_id]
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Clear chat history for a session
    
    Args:
        session_id: The session ID
    """
    if session_id in session_history:
        del session_history[session_id]
        return {"status": "success", "message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "total_sessions": len(session_history),
        "sessions": [
            {
                "session_id": sid,
                "message_count": len(history)
            }
            for sid, history in session_history.items()
        ]
    }


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return {
        "status": "error",
        "message": str(exc)
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("Starting College Buddy Backend Server...")
    print("=" * 70)
    print("\nServer will be available at: http://127.0.0.1:8000")
    print("API Documentation: http://127.0.0.1:8000/docs")
    print("ReDoc Documentation: http://127.0.0.1:8000/redoc")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    # Run with uvicorn
    uvicorn.run(
        "backend:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )
