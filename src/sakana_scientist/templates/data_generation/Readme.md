## The examples, prompts and rubric are borrowed from ARENA for this PoC

Since we're using LLMs to generate questions, you'll need to be able to clearly explain to the LLM what sorts of questions you want to be generated. Thus, to do the exercises in this section, you will need (at minimum):
 - A good understanding of the target property you want to measure
 - A working definition you can give to LLMs for this target property
 - Some high-quality example questions which evaluate this target property effectively (you may be able to use LLMs to help you generate some of these)
 - Some mediocre and bad questions that aren't as good at measuring this target property (this is particularly important when we use LLMs to score the questions we've generated, so we can show our scoring LLM what bad and mediocre questions look like, as compared to high-quality questions)


## Steps

1. Setup the MCQ structures for GPT-4o calls in structured_outputs.py
2. Write prompts and all config needed to hydrate prompts for dataset generation in prompt.json
3. Collect few shot examples in few-shot-examples.json
4. Provide a rubric that illustrates the scoring mechanism
5. Create do_dataset_generation() function