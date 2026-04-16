import httpx
from pydantic import BaseModel, HttpUrl
from typing import Optional, List

class ExtractRequest(BaseModel):
    urls: Optional[List[HttpUrl]] = None
    
async def resolve_url(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, timeout=10)
            return str(response.url)
    except Exception as e:
        print(f"Erro ao resolver {url}: {e}")
        return None