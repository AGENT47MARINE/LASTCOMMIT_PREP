from solver import MathSolver

def main():
    solver = MathSolver()
    query = (
        "How many distinct 4x4 Latin squares are there? "
        "(A Latin square of order n is an nxn array filled with n different symbols, "
        "each occurring exactly once in each row and exactly once in each column.)"
    )
    
    answer = solver.solve(query)
    print(answer)

if __name__ == "__main__":
    main()
