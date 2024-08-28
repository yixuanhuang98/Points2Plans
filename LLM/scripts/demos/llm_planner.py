from typing import Union, Optional

import ast
import argparse
import pathlib
import sys
import os
sys.path.append(os.getcwd())
from LLM.fm_planning import models, utils


def demo(
    model_config: Union[str, pathlib.Path],
    prompt_config: Union[str, pathlib.Path],
    api_key: Optional[str] = None,
    device: str = "auto",
    seed: int = 0,
) -> None:
    """Run a demo."""
    # Set seed.
    utils.random.seed(seed)

    # Load model.
    model_factory = models.PretrainedModelFactory(
        model_config, api_key=api_key, device=device
    )
    model = model_factory()

    # Load prompt.
    prompt_manager = models.BehaviorPromptManager.from_yaml(prompt_config)

    # Predict goals.
    goal_prediction_prompt = prompt_manager.generate_prompt(
        behavior="goal_prediction",
        use_examples=True,
    )
    response = model.forward(goal_prediction_prompt)
    predicted_goals = ast.literal_eval(response["choices"][0]["message"]["content"])

    # Update prompts with predicted goals.
    prompt_manager.task_prompt.goals = predicted_goals

    # Predict plans.
    task_planning_prompt = prompt_manager.generate_prompt(
        behavior="task_planning",
        use_examples=True,
    )
    response = model.forward(task_planning_prompt)
    predicted_plans = ast.literal_eval(response["choices"][0]["message"]["content"])

    print(f"Predicted goals: {predicted_goals}")
    print(f"Predicted plans: {predicted_plans}")


def main(args: argparse.Namespace) -> None:
    demo(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--prompt-config", type=str, required=True)
    parser.add_argument("--api-key", type=str)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
