from llm import get_response_from_llm, extract_json_between_markers, create_client, AVAILABLE_LLMS
import json
import os
import os.path as osp
import time
from typing import List, Dict, Union
import datetime

from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate
from inspect_ai.model import get_model

@task
def basic_eval(dataset: Dataset, scorer: str):
    # Load the dataset from Hugging Face with the specified parameters

    # model = get_model("hf/openai-community/gpt2", device="cuda:0")

    # Define the task with the dataset, solver, and scorer
    return Task(
        dataset=dataset,
        solver=[generate()],  # Use the generate solver to produce outputs
        scorer=model_graded_fact(model='openai/gpt-4o')  # Use a model-based scorer for evaluation
    )

def generate_basic_eval(args, base_dir: str, client, model: str,):
    # Fetch the config for this template
    # TODO: Read config from json file
    config = json.load(open(osp.join(base_dir, "config.json")))

    # Generate the dataset
    dataset = hf_dataset(
        dataset=config["dataset"],
        split=config["dataset_split"],
        limit=config["limit"],
        trust=config["trust"]
    )
    
    


if __name__ == "__main__":
    import argparse
    from llm import AVAILABLE_LLMS

    parser = argparse.ArgumentParser(description="Generate AIS scientist evals")
    # add type of dataset 
    parser.add_argument("--experiment",type=str,default="evals",help="Generate evals for AIS Scientists.")
    parser.add_argument("--model",type=str,default="gpt-4o-2024-08-06",choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.")
    
    args = parser.parse_args()

    # Create client
    client, client_model = create_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)
    dataset = generate_basic_eval(args,
        base_dir,
        client=client,
        model=client_model,
        
    )

