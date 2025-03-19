import json
import os
import os.path as osp
import time
from typing import List, Dict, Union

import backoff
import requests

from llm import get_response_from_llm, extract_json_between_markers, create_client, AVAILABLE_LLMS

from data_generation.structured_outputs import hydrate_prompts, QuestionGeneration, MCQs
import datetime


def generate_dataset(
    base_dir: str,
    client,
    model: str,
    max_num_generations: int,
) -> List[Dict[str, Union[str, Dict[str, str]]]]:
    
    system_prompt, user_prompt = hydrate_prompts(base_dir, few_shot=False)

    # Generate dataset
    print(f"Generating dataset")

    msg_history = []
    text, msg_history = get_response_from_llm(
        user_prompt,
        client=client,
        model=model,
        system_message=system_prompt,
        msg_history=msg_history,
        response_format=MCQs
    )
    ## PARSE OUTPUT
    # json_output = extract_json_between_markers(text)
    # assert json_output is not None, "Failed to extract JSON from LLM output"
    # print(json_output)

    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"{base_dir}/output_{model}_{timestr}.json", 'w') as f:
        json.dump(text, f)



if __name__ == "__main__":
    MAX_NUM_GENERATIONS = 1
    import argparse

    parser = argparse.ArgumentParser(description="Generate AIS scientist datasets")
    # add type of dataset 
    parser.add_argument(
        "--experiment",
        type=str,
        default="data_generation",
        help="Generate datasets for AIS Scientists.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-08-06",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    args = parser.parse_args()

    # Create client
    client, client_model = create_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)
    dataset = generate_dataset(
        base_dir,
        client=client,
        model=client_model,
        max_num_generations=MAX_NUM_GENERATIONS,
    )

    # TODO: 
    # 1. Using the rubric provided, score the dataset generated. 
    # 2. Discard datapoints scored below a certain threshold.
    # 3. If n_reflections > 0: iteratively repeat this process
    # 
