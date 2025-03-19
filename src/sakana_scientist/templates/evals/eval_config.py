import argparse

parser = argparse.ArgumentParser(description="Generate AIS scientist evals")
# add type of dataset 
parser.add_argument("dataset", type=str, default="HuggingFaceH4/ultrachat_200k", help="Path to dataset")
parser.add_argument("dataset_split", type=str, default="validation", help="dataset split to evaluate over")
parser.add_argument("limit", type=int, default=10, help="Inspect_Evals: Number of examples to evaluate")
parser.add_argument("trust", type=bool, default=False, help="True if downloading dataset to local")
parser.add_argument("scorer", type=str, default="model_graded_fact", help="Inspect_Evals: Scorer to use")
parser.add_argument("scorer_model", type=str, default="openai/gpt4o", help="Inspect_Evals: Model for Scorer")


args = parser.parse_args()
