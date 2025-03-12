from inspect_ai.dataset import Sample, json_dataset
import json

# Define the setup we would need to provide to an LLM to generate the evaluation code for us

config = {
"dataset_config": { "path": "HuggingFaceH4/ultrachat_200k",
  "split": "test_sft",
  "limit": 10,
  "trust": True,
  "field_mapping": {"input":"prompt", "target":"messages[assistant]", }
},
"scorer": {"scorer_type":"model_graded_fact", "model":"openai/gpt-4o"},
"task_id": "ultrachat_eval" 
}

SYSTEM_PROMPT = f""" 
Given the dataset, field mapping and scorer, generate a Task object using the Inspect Evaluation Framework.
You will be rated on how accurate your generated code is. Pay attention to the details of the Inspect library documentation. 
Return python code only, no explanations. 
""" 
USER_PROMPT = f"""Create an inspect task using the config provided below.
If the dataset has a hugging face location use the hugging face dataset loader from Inspect. 
If the split is empty, use the default split with a default of 100 records. 
Use the field mapping provided to generate a record_to_sample function. If the task id is not provided, use an id. 
If there are extra columns, load into the metadata dictionary"""
USER_PROMPT_CFG = "The relevant config is here '{config}'"

from utils.utils import call_llm, code_llm

if __name__ == "__main__":

    # response = call_llm("gpt-4o", SYSTEM_PROMPT, USER_PROMPT+USER_PROMPT_CFG.format(config=config), verbose=True) 
    response = code_llm("gpt-4o", SYSTEM_PROMPT, USER_PROMPT+USER_PROMPT_CFG.format(config=config), verbose=True)
    print(response)
