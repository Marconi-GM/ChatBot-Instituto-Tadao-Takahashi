from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  
from fastapi.responses import JSONResponse
import uvicorn

from agent import ITTAgent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = ITTAgent()

@app.post("/query_response")
async def query_response(request: dict):

    response = agent.process_query(request)

    return JSONResponse(status_code=200, content={"response": response})

if __name__ == "__main__":
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )