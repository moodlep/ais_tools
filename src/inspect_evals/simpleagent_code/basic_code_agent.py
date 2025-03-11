
from inspect_ai.dataset import Sample
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import basic_agent, system_message
from inspect_ai.tool import bash, python
from inspect_ai.scorer import includes

# Read in the Dataset:
def record_to_sample(record):
    return Sample(
        input=record["input"],
        target=record["output"],
        id=record["task"],
        # metadata={"test_input": record["testcase"]["input"], "test_output": record["testcase"]["output"]}
    )

# Solver is a basic agent running in a docker container
SYSTEM_TESTS = """ You are a coding agent. You will be given a set of simple coding tasks. Please code this in python in the machine provided. 
For each task there is a test so you need to code the task, then using the input provided, run the test and return only the answer. 
"""

dataset = json_dataset("datasets.json", record_to_sample)
@task
def code_agent():
    return Task(dataset,
                solver=basic_agent(
                    init=system_message(SYSTEM_TESTS),
                    tools=[bash(timeout=180), python(timeout=180)],
                    max_attempts=1,
                    message_limit=50,
                ),
                scorer=includes(),
                sandbox="docker",
                )

