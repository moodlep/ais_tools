import json
import os
import openai
from openai import OpenAI
import dotenv
import json
import yaml
from openai import OpenAI
import types
from typing import List, Optional, Protocol, Literal, Callable, Dict, Any, Tuple, ClassVar
import warnings
from pydantic import BaseModel
from dataclasses import dataclass, field, asdict, fields
import random
import time

# ***** ARENA *****
MODEL = "gpt-4o"
MAX_NUM_TOKENS = 1000

def retry_with_exponential_backoff(
    func, retries=20, intial_sleep_time: int = 3, jitter: bool = True, backoff_factor: float = 1.5
):
    """
    This is a sneaky function that gets around the "rate limit error" from GPT (GPT has a maximum tokens per min processed, and concurrent processing may exceed this) by retrying the model call that exceeds the limit after a certain amount of time.
    """

    def wrapper(*args, **kwargs):
        sleep_time = intial_sleep_time
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate_limit_exceeded" in str(e):
                    sleep_time *= backoff_factor * (1 + jitter * random.random())
                    time.sleep(sleep_time)
                else:
                    raise
        raise Exception(f"Maximum retries {retries} exceeded")

    return wrapper

def apply_system_format(content: str) -> dict:
    return {"role": "system", "content": content}


def apply_user_format(content: str) -> dict:
    return {"role": "user", "content": content}


def apply_assistant_format(content: str) -> dict:
    return {"role": "assistant", "content": content}


def apply_message_format(user: str, system: str | None) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages

@retry_with_exponential_backoff
def generate_formatted_response(client: OpenAI,
                                model = MODEL,
                                messages:Optional[List[dict]]=None,
                                user:Optional[str]=None,
                                system:Optional[str]=None,
                                temperature:int=1,
                                verbose: bool = False,
                                response_format:BaseModel=None) -> Dict[str, Any]:
    '''
    Generate a formatted response using the OpenAI API `client.beta.chat.completions.parse()` function.

    Args:
        client (OpenAI): OpenAI API client.
        messages (Optional[List[dict]], optional): List of formatted message dictionaries with 'role' and 'content' keys.
        user (Optional[str], optional): User message string (alternative to `messages`).
        system (Optional[str], optional): System message string (alternative to `messages`).
        verbose (bool, optional): If True, prints detailed message information. Defaults to False.

    Returns:
        dict: The model response as a dictionary that contains a "rt_prompts" key, which stores the list of generated red teaming prompts.

    Note:
        - If both `messages` and `user`/`system` are provided, `messages` takes precedence.

    '''
    if messages is None:
        messages = apply_message_format(user=user, system=system)

    if verbose:
        for message in messages:
            print(f"{message['role'].upper()}:\n{message['content']}\n")

    if response_format is None:

        completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=MAX_NUM_TOKENS,
                    )
        response = completion.choices[0].message

    else:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            response_format=response_format,
        )

    response = completion.choices[0].message

    return response

def call_llm(model, system_prompt, user_prompt, verbose=False):
    
    # Configure your OpenAI API key
    dotenv.load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key
    client = OpenAI()

    response = generate_formatted_response(client=client, model=model, system=system_prompt, user=user_prompt, messages=None, verbose=verbose, response_format=None)

    return response        