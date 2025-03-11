from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate
from inspect_ai.model import get_model
from inspect_ai.dataset import Sample, json_dataset

@task
def ultrachat_eval():
    # Load the dataset from Hugging Face with the specified parameters
    dataset = hf_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="test_sft",  # Adjust split if needed (e.g., "train" or "test")
        limit=10,          # Limit to 1000 records
        trust=True           # Trust flag for downloading the dataset
    )

    # Define the task with the dataset, solver, and scorer
    return Task(
        dataset=dataset,
        solver=[generate()],  # Use the generate solver to produce outputs
        scorer=model_graded_fact(model='openai/gpt-4o')  # Use a model-based scorer for evaluation
    )
