#!/usr/bin/env python3
"""
FastAPI Backend Server for College Chatbot
Provides REST API endpoints for the frontend to interact with UltraRAG
"""

import sys
from pathlib import Path
import uuid
import json
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time
from contextlib import asynccontextmanager

# Add app directory to path
# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.query_router import QueryRouter
from app.services.logger_service import ResponseLogger
from app.services.embedding_service import get_embedding_model

# Initialize global logger
response_logger = None


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

from pydantic import Field

class QueryRequest(BaseModel):
    """Request model for chat queries"""
    message: str = Field(..., max_length=1000, description="The user's chat query")
    session_id: Optional[str] = None
    language: Optional[str] = "en"


class QueryResponse(BaseModel):
    """Response model for chat queries"""
    answer: str
    session_id: str
    source: Optional[str] = "Unknown"
    confidence: Optional[int] = None
    time_taken: Optional[float] = None


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
    
    # Warm-up Ollama in background to reduce first-query latency
    if query_router:
        print("Pre-loading Ollama model...")
        try:
            import threading
            def warmup():
                try:
                    query_router.rag_system.generator.generate(
                        query="warmup", docs=[], kb_context="", links=[], is_greeting=True
                    )
                    print("✓ Ollama model warmed up!")
                except Exception as e:
                    print(f"⚠ Warmup failed: {e}")
            
            threading.Thread(target=warmup, daemon=True).start()
        except Exception as e:
            print(f"⚠ Could not start warmup thread: {e}")
    
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
    allow_origins=["*"],  # Allow all origins for local development and file:// access
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
session_timeout_count = {}  # Track timeout retries per session


def initialize_systems():
    """Initialize Query Router and subsystems"""
    global query_router, response_logger
    print("\n" + "=" * 70)
    print("COLLEGE BUDDY - BACKEND SERVER")
    print("=" * 70)
    print("\nInitializing Systems...")
    try:
        response_logger = ResponseLogger()
        print(f"✓ Response Logger initialized. Saving logs to: {response_logger.log_file}")
        query_router = QueryRouter()
        print("✓ Query Router & Unified Brain initialized successfully!")
        return True
    except Exception as e:
        print(f"✗ Error initializing systems: {e}")
        print("Make sure:")
        print("  1. All data files are present")
        print("  2. Ollama is running: ollama serve")
        print("  3. Model is pulled: ollama pull llama3.2:3b")
        return False


 


# ============================================================
# API ENDPOINTS
# ============================================================

def log_async(user_query, bot_response, time_taken, session_id, source, accuracy_label=None, context_snippets=None):
    """Background task to log the response to the Production sheet."""
    try:
        if response_logger:
            response_logger.log_response(
                user_query=user_query,
                bot_response=bot_response,
                time_taken=time_taken,
                session_id=session_id,
                source=source,
            )
        print(f"✓ Background logging complete for session: {session_id}")
    except Exception as e:
        print(f"⚠ Background logging failed: {e}")


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


@app.post("/query")
async def chat(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Main chat endpoint - accepts user queries and returns responses
    Supports streaming for RAG queries to improve perceived latency.
    
    Request:
        - message: User's question
        - session_id: Optional session ID for conversation continuity
    
    Response:
        - For streaming (RAG): Server-Sent Events stream
        - For non-streaming (SQL): JSON response with answer and session_id
    """
    from fastapi.responses import StreamingResponse
    from app.services.intent_detector import IntentDetector
    
    if query_router is None:
        raise HTTPException(
            status_code=503, 
            detail="System not initialized. Please check the server logs and ensure all dependencies are installed."
        )
    
    # Validate and sanitize message
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    # Basic sanitization: strip excessive whitespace
    import re
    cleaned_message = re.sub(r'[\r\n\t]+', ' ', request.message.strip())
    request.message = cleaned_message
    
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
        
        # Prepare chat history context (last 3 turns)
        recent_history = session_history[session_id][-6:] if len(session_history[session_id]) > 0 else []
        
        # Detect intent to decide streaming vs non-streaming
        # Reuse the global router's detector (loaded once at startup)
        intent = query_router.intent_detector.detect_intent(request.message)
        
        # SQL queries are fast, no need to stream
        # RAG queries benefit from streaming
        use_streaming = (intent == 'general' or intent == 'hybrid')
        
        start_time = time.time()
        
        if use_streaming:
            # Streaming response for RAG queries
            async def generate_stream():
                """Generator for streaming response - accumulates chunks for logging"""
                try:
                    # Get the RAG system directly
                    rag_system = query_router.rag_system
                    
                    actual_source = "RAG (Streamed)" # Default
                    confidence = "N/A"
                    context_snippets = []
                    full_response = ""  # Accumulate complete response for logging
                    
                    # Stream the response
                    for chunk in rag_system(request.message, stream=True):
                        # Check for metadata
                        if isinstance(chunk, dict) and chunk.get("type") == "metadata":
                            actual_source = chunk.get("source", actual_source)
                            confidence = chunk.get("confidence", "N/A")
                            context_snippets = chunk.get("context", [])
                            # Forward metadata to client so test scripts can see it
                            yield f"data: {json.dumps(chunk)}\n\n"
                            continue
                        
                        # Accumulate the chunk for logging
                        full_response += chunk
                            
                        # Format as Server-Sent Events
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    
                    # Send completion signal with server-side time_taken
                    end_time = time.time()
                    response_time = round(end_time - start_time, 4)
                    yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'time_taken': response_time})}\n\n"
                    
                    # Log the COMPLETE response after streaming finishes in background
                    if response_logger:
                        background_tasks.add_task(
                            log_async,
                            request.message,
                            full_response,
                            response_time,
                            session_id,
                            actual_source if "actual_source" in locals() else "RAG (Streamed)",
                        )
                except Exception as e:
                    print(f"Streaming error: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Disable nginx buffering
                }
            )
        else:
            # Non-streaming response for SQL queries (they're already fast)
            response_source = "Unknown"
            response_accuracy = "N/A"
            
            try:
                import requests
                from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
                
                # Run query_router in a thread with timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(query_router.route_query, request.message, recent_history)
                    try:
                        result = future.result(timeout=70)  # 70 second timeout
                        
                        # Handle dictionary response (new format)
                        if isinstance(result, dict):
                            answer = result.get('response', str(result))
                            response_source = result.get('source', 'Unknown')
                            response_accuracy = result.get('accuracy', 'N/A')
                        else:
                            # Fallback for old format
                            answer = str(result)
                        
                        # Reset timeout count on success
                        if session_id in session_timeout_count:
                            session_timeout_count[session_id] = 0
                    except FuturesTimeoutError:
                        # Track timeout attempts
                        session_timeout_count[session_id] = session_timeout_count.get(session_id, 0) + 1
                        
                        if session_timeout_count[session_id] == 1:
                            answer = "I'm taking a bit too long to think about that. Could you try asking in a simpler way?"
                        else:
                            answer = "I apologize, but I'm having trouble processing this question right now. Please try a different question or contact the college office for immediate assistance."
                            session_timeout_count[session_id] = 0
                        
                        response_source = "System"
                        response_accuracy = "Timeout"
            except Exception as query_error:
                print(f"Query execution error: {query_error}")
                answer = "I encountered an error while processing your question. Please try again or rephrase your question."
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Store bot response in history
            session_history[session_id].append({
                "role": "bot",
                "message": answer
            })
            
            # Offload heavy metrics and logging to background
            if response_logger:
                background_tasks.add_task(
                    log_async,
                    request.message,
                    str(answer),
                    response_time,
                    session_id,
                    response_source,
                    response_accuracy
                )
            
            # Keep session history manageable (last 50 exchanges)
            if len(session_history[session_id]) > 100:
                session_history[session_id] = session_history[session_id][-100:]
            
            return QueryResponse(
                answer=str(answer),
                session_id=session_id,
                source=response_source,
                confidence=100 if response_accuracy == "High" else 50 if response_accuracy == "N/A" else 0,
                time_taken=round(response_time, 4)
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
