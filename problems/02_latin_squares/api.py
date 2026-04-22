from fastapi import FastAPI
from solver import MathSolver
from pydantic import BaseModel

app = FastAPI()
solver = MathSolver()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Math Solver API is online"}

@app.post("/v1/answer")
def get_answer(request: QueryRequest):
    answer = solver.solve(request.query)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
