from .solver import QAPracticeButtonSolver


def main():
    solver = QAPracticeButtonSolver()
    query = (
        "Go to the link and find the answer for the task given task. "
        "Step 1 : Go to the link and click on the simple button tab. "
        "Step 2 : Click on the button named 'Click'. "
        "Step 3 : A confirmation message appears in a box, that is your answer. "
        "Step 4 : Return the answer."
    )
    assets = ["https://www.qa-practice.com/elements/button/simple"]
    answer = solver.solve(query, assets)
    print(answer)


if __name__ == "__main__":
    main()

