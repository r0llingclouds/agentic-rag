import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import json
from crew_service import crew_chain
import asyncio



router = APIRouter()

class QueryRequest(BaseModel):
    user_query: str

async def stream_crew_output(query):
    try:
        # Run CrewAI in a separate thread
        result = await asyncio.to_thread(crew_chain.kickoff, {"query": query})

        # Ensure TaskOutput is converted properly
        if hasattr(result, "raw"):  
            response_text = result.raw  # Extract raw text
        elif hasattr(result, "__dict__"):  
            response_text = json.dumps(result.__dict__)  # Convert TaskOutput to JSON
        else:
            response_text = str(result)  # Fallback: Convert to string

        # Simulate streaming by splitting response into chunks
        for chunk in response_text.split(". "):  
            yield json.dumps({"message": chunk.strip() + "."})  
            await asyncio.sleep(1)  

    except Exception as e:
        yield json.dumps({"error": str(e)})


@router.post("/query")
async def run_crew(request: QueryRequest):
    try:
        #result = await crew_chain.kickoff({"query": request.user_query})
        result = await asyncio.to_thread(crew_chain.kickoff, {"query": request.user_query})

        #result=json.loads(result)
        # # Convert CrewOutput object to dictionary
        # if hasattr(result, "__dict__"):  
        #     result = result.__dict__  

        # # Ensure result is a dictionary
        # if not isinstance(result, dict):
        #     raise ValueError("Unexpected response format from CrewAI.")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    return {
        "answer": result.raw
    }
    
