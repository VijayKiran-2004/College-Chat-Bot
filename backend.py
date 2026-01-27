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

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.ultra_rag import UltraRAGSystem


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

app = FastAPI(
    title="College Buddy Backend",
    description="REST API for College Chatbot using UltraRAG",
    version="1.0.0"
)

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

# Initialize UltraRAG system on startup
rag_system = None
session_history = {}  # Store chat history by session_id


def initialize_rag():
    """Initialize UltraRAG system"""
    global rag_system
    print("\n" + "=" * 70)
    print("COLLEGE BUDDY - BACKEND SERVER")
    print("=" * 70)
    print("\nInitializing UltraRAG System...")
    try:
        rag_system = UltraRAGSystem()
        print("✓ UltraRAG System initialized successfully!")
        return True
    except Exception as e:
        print(f"✗ Error initializing UltraRAG: {e}")
        print("Make sure:")
        print("  1. All data files are present in app/database/vectordb/")
        print("  2. Ollama is running: ollama serve")
        print("  3. Model is pulled: ollama pull gemma2:2b")
        return False


@app.on_event("startup")
async def startup_event():
    """Run on server startup"""
    initialize_rag()
    # Don't raise exception - allow server to start even if RAG initialization has issues
    # The error messages will guide the user on what to fix


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
    if rag_system is None:
        raise HTTPException(status_code=503, detail="UltraRAG system not initialized")
    
    return {
        "status": "healthy",
        "rag_system": "ready",
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
    
    if rag_system is None:
        raise HTTPException(
            status_code=503, 
            detail="UltraRAG system not initialized. Please check the server logs and ensure all dependencies are installed."
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
        
        # Generate response using UltraRAG
        answer = rag_system(request.message)
        
        # Store bot response in history
        session_history[session_id].append({
            "role": "bot",
            "message": answer
        })
        
        # Keep session history manageable (last 50 exchanges)
        if len(session_history[session_id]) > 100:
            session_history[session_id] = session_history[session_id][-100:]
        
        return QueryResponse(
            answer=answer,
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
