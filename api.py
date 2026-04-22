from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from problems.p04_qa_practice_button.solver import QAPracticeButtonSolver

app = FastAPI()
solver = QAPracticeButtonSolver()


class QueryRequest(BaseModel):
    query: str
    assets: Optional[List[str]] = Field(default_factory=list)


@app.get("/")
def read_root():
    return {"message": "LastCommit Prep API is online"}


@app.post("/v1/answer")
def get_answer(request: QueryRequest):
    answer = solver.solve(request.query, request.assets)
    return {"output": answer}

