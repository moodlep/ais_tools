from inspect import Task
from inspect.data.huggingface import HuggingFaceDatasetLoader
from inspect.scorers import ModelGradedFactScorer



# Define the record_to_sample function based on field mapping

def record_to_sample(record):

    sample = {

         'input': record['prompt'],

         'target': record['messages']['assistant']
    }
    # Add additional columns to metadata if they exist

    metadata = {key: value for key, value in record.items() if key not in ['prompt', 'messages']}

    sample['metadata'] = metadata
    return sample

# Create Task object using the Inspect Evaluation Framework
task = Task(
    task_id='ultrachat_eval',  # Given task id
    dataset_loader=HuggingFaceDatasetLoader(
        path='HuggingFaceH4/ultrachat_200k',
        split='test_sft',  # Use specified split
        limit=10,  # Use specified limit
        trust=True
    ),
    scorer=ModelGradedFactScorer(
        model='openai/gpt-4o'
    ),
    record_to_sample=record_to_sample
)
