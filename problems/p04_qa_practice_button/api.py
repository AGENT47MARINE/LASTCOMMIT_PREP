from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .solver import QAPracticeButtonSolver

app = FastAPI()
solver = QAPracticeButtonSolver()


class QueryRequest(BaseModel):
    query: str
    assets: Optional[List[str]] = Field(default_factory=list)


@app.get("/")
def read_root():
    return {"message": "QA Practice Button Solver is online"}


@app.post("/v1/answer")
def get_answer(request: QueryRequest):
    answer = solver.solve(request.query, request.assets)
    return {"output": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

