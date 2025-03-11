from inspect_ai import task, Task
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import chain_of_thought, generate

@task
def theory_of_mind():
    return Task(
        dataset=example_dataset("theory_of_mind")[0:3],
        plan=[
            chain_of_thought(),
            generate(),
        ],
        scorer=model_graded_fact(),
    )