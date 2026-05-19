"""HTTP server for lanka_data.

Usage:
    uvicorn workflows.server:app --reload

Then query:
    GET /query/{where}/{what}/{when}
    e.g. http://localhost:8000/query/LK/population/2012
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from lanka_data import Db

app = FastAPI(title="lanka_data API", version="1.0.0")


@app.get("/query/{where}/{what}/{when}")
def query(where: str, what: str, when: str):
    path = f"/{where}/{what}/{when}"
    try:
        result = Db(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(content=result)
