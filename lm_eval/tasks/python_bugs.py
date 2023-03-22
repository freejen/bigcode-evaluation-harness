"""Python Bugs
https://proceedings.mlr.press/v162/he22a.html

This dataset is taken from the preprossing done by CarperAI (https://carper.ai/diff-models-a-new-way-to-edit-code).
It is uploaded here: https://huggingface.co/datasets/Muennighoff/python-bugs

Make sure to run with sufficient context length (512 is not enough for e.g. CodeGen).
"""

import re
from evaluate import load
from lm_eval.base import Task
import tqdm

_CITATION = """
@inproceedings{he2022distribution,
  title={On distribution shift in learning-based bug detectors},
  author={He, Jingxuan and Beurer-Kellner, Luca and Vechev, Martin},
  booktitle={International Conference on Machine Learning},
  pages={8559--8580},
  year={2022},
  organization={PMLR}
}
"""

MUTATE_TO_TASK_TO_PROMPT = {
    "prompt": {
        "bin-op": "# Fix binary operator",
        "var-misuse": "# Fix incorrect variable name",
    },
    "edit": {
        "bin-op": "Fix binary operator",
        "var-misuse": "Fix incorrect variable name",
    },
}

def mutate_code(input_code, task, mutate_method="prompt"):
    """
    Create template for code mutation.
    Args:
        input_code: code to be mutated
        task: task to be performed
        mutate_method: (Optional) 'edit' or 'prompt'
    Returns:
        template for code mutation
    """
    instruction = MUTATE_TO_TASK_TO_PROMPT[mutate_method][task]
    if mutate_method == "prompt":
        return f"{input_code}\n{instruction}\n"
    if mutate_method == "edit":
        return f"<commit_before>{input_code}<commit_msg>{instruction}<commit_after>"
    else:
        raise ValueError(f"Unknown mutate_method: {mutate_method}")



class PythonBugs(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "Muennighoff/python-bugs"

    def __init__(self):
        super().__init__(
            # Correct code always starts with `def ...` and is a single function, so stop everything else
            # Since a function always has a tab, stop when the first line does not have a tab
            stop_words=["\nclass", "\n#", "\ndef", "\nassert", '\n"', "\nprint", "\nif"],
            requires_execution=True,
        )
        self.max_length_multiplier = 2.25 # Allow 2.25 times the length of the prompt
        self.mutate_method = "prompt"

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["train"]
        return dataset

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return mutate_code(doc["prompt_code"], doc["task"], self.mutate_method)

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["correct_code"]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        doc = self.get_dataset()[idx]
        prompt = self.get_prompt(doc)
        correct_code = self.get_reference(doc)
        output = generation[len(prompt):]
        output = output[:len(correct_code)]
        return output

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        num_correct = 0
        print("Scoring generations...")
        for i, ref in tqdm.tqdm(enumerate(references), total=len(references)):
            for gen in generations[i]:
                num_correct += int(gen == ref)
        accuracy = num_correct / len(references) / len(generations[0])
        return {f"mean exact match ({len(generations[0])} samples)": accuracy}