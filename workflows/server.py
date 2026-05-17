"""HTTP server for lanka_data.

Usage:
    uvicorn workflows.server:app --reload

Then query:
    GET /query/{what}/{when}/{where}
    e.g. http://localhost:8000/query/population/2012/LK
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from lanka_data import Db

app = FastAPI(title="lanka_data API", version="1.0.0")


@app.get("/query/{what}/{when}/{where}")
def query(what: str, when: str, where: str):
    path = f"/{what}/{when}/{where}"
    try:
        result = Db(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(content=result)
