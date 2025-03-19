from pydantic import BaseModel
from typing import List, Dict, Optional, Literal
import json
import os
import os.path as osp

class Answer(BaseModel):
    A: str
    B: str

class MCQ(BaseModel):
  system: str
  question: str
  answers: Answer
  answer_matching_behavior: List[str]
  answer_not_matching_behavior: List[str]
  question_category: str

class MCQs(BaseModel):
  questions: List[MCQ]

# class Question(BaseModel):
#     system: str
#     question: str
#     answers: Answer
#     answer_matching_behavior: list[Literal["A", "B"]]
#     answer_not_matching_behavior: list[Literal["A", "B"]]
#     behavior_category: str

# class QuestionGeneration(BaseModel):
#     reasoning: str  # Allow model to do chain-of-thought reasoning before generating the questions
#     questions: List[Question]


def hydrate_prompts(base_dir: str, few_shot: Optional[bool] = True):
  # Scaffolding code to read from prompt.json, collect samples from few-shot-samples.json
    # Load prompts and configurations
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompts = json.load(f)

    # hydrate user prompt
    user_prompt = prompts["user_prompt"].format(num_q_per_call=prompts['num_q_per_call'], evaluation_target=prompts['evaluation_target'], 
                                     target_definition=prompts['target_definition'], 
                                     good_question_description=prompts['good_question_description'], 
                                     extra_instruction=prompts['extra_instruction'])
    
    if few_shot:
        user_prompt = add_few_shot(base_dir, user_prompt)
    
    print(user_prompt)

    return prompts['system_prompt'], user_prompt

def add_few_shot(base_dir: str, user_prompt: str):

    # Load few-shot examples
    with open(osp.join(base_dir, "few_shot_samples.json"), "r") as f:
        few_shot = json.load(f)

    # Add few-shot examples to user prompt
    user_prompt += "\n\nHere are some examples of questions and answers:\n"
    for example in few_shot:
        user_prompt += f"\n{example}"
    return user_prompt

if __name__ == "__main__":

  hydrate_prompts("templates/data_generation")
