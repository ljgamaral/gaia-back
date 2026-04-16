from fastapi import FastAPI
from extraction import load_urls, extract_many, save_articles
from typing import Optional, List
from pydantic import HttpUrl, BaseModel
from fastapi.middleware.cors import CORSMiddleware
from label_sentiment import label_row

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExtractRequest(BaseModel):
    urls: Optional[List[HttpUrl]] = None
    urls_file: str = "urls.txt"
    out: str = "extracted_urls.json"
    continue_on_error: bool = True
    timeout_seconds: float = 10
    max_workers: int = 20
    
class AnalyzeRequest(BaseModel):
    type: str 
    content: str
    
@app.post("/extract") 
def extract(request: ExtractRequest) -> dict:
    urls = load_urls(urls=request.urls, urls_file=request.urls_file)
    articles = extract_many(
        urls,
        request.continue_on_error,
        request.timeout_seconds,
        request.max_workers,
    )
    save_articles(articles, request.out)
    return {"count": len(articles), "out": request.out, "articles": articles}

@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    if request.type == "link":
        try:
            content = extract_many([request.content], continue_on_error=False, timeout_seconds=5, max_workers=1)
            label, reason = label_row(content[0])
        except Exception as e:
            return {"success": False, "error": str(e)}
        return {
            "success": True,
            "content": {
                "label": label,
            }
        }
        
    elif request.type == "text":
        return {"message": f"Analisando o texto: {request.content}"}