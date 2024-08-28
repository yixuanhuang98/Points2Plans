from typing import Union, List, Dict, Optional, Any, Tuple, Set

import yaml
import pathlib
import dataclasses


def format_openai(
    role: str,
    content: str,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Format a prompt for OpenAI."""
    prompt = {"role": role, "content": content}
    if name is not None:
        prompt["name"] = name
    return prompt


@dataclasses.dataclass
class SystemPrompt:
    content: str
    # Behavior
    behavior: str
    behavior_kwargs: Optional[Dict[str, Any]] = None
    # OpenAI
    role: Optional[str] = None
    name: Optional[str] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SystemPrompt":
        """Create a behavior prompt from a config."""
        return cls(**config)

    @classmethod
    def from_yaml(cls, file: Union[str, pathlib.Path]) -> "SystemPrompt":
        """Create a behavior prompt from a YAML file."""
        if isinstance(file, str):
            file = pathlib.Path(file)

        with file.open("r") as f:
            config = yaml.safe_load(f)

        return cls.from_config(config)

    def prompt(self, openai: bool = True) -> Union[str, Dict[str, Any]]:
        """Return the prompt."""
        if openai:
            assert self.role is not None
            prompt = format_openai(role=self.role, content=self.content, name=self.name)
        else:
            prompt = self.content

        return prompt


@dataclasses.dataclass
class BehaviorPrompt:
    # Task
    task: Optional[str] = None
    instruction: Optional[str] = None
    objects: Optional[List[str]] = None
    predicates: Optional[List[str]] = None
    # Behaviors
    goals: Optional[List[List[str]]] = None
    plans: Optional[List[List[str]]] = None
    hypotheses: Optional[List[List[str]]] = None
    geometric: Optional[bool] = None
    # OpenAI
    role: Optional[str] = None
    name: Optional[str] = None
    name_query: Optional[str] = None
    name_response: Optional[str] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BehaviorPrompt":
        """Create a behavior prompt from a config."""
        return cls(**config)

    @classmethod
    def from_yaml(cls, file: Union[str, pathlib.Path]) -> "BehaviorPrompt":
        """Create a behavior prompt from a YAML file."""
        if isinstance(file, str):
            file = pathlib.Path(file)

        with file.open("r") as f:
            config = yaml.safe_load(f)

        return cls.from_config(config)

    def _prefix(self) -> str:
        """Return the prefix of the task prompt."""
        assert (
            self.objects is not None
            and self.predicates is not None
            and self.instruction is not None
        )
        prefix = ""
        prefix += f"Objects: {self.objects}\n\n"
        prefix += f"Object Relationships: {self.predicates}\n\n"
        prefix += f"User Instruction: {self.instruction}"
        return prefix

    def _goal_prediction(self) -> Tuple[str, Optional[str]]:
        """Return the goal prediction prompt."""
        query = self._prefix() + "\n\n" + "Goals: "
        response = str(self.goals) if self.goals else None
        return query, response

    def _task_planning(self, goal_condition: bool = False) -> Tuple[str, Optional[str]]:
        """Return the task planning prompt."""
        if goal_condition:
            query, goals = self._goal_prediction()
            assert goals is not None
            query += goals + "\n\n" + "Plans: "
        else:
            query = self._prefix() + "\n\n" + "Plans: "

        response = str(self.plans) if self.plans else None
        return query, response

    def _hypothesis_generation(self) -> str:
        """Return the hypothesis generation prompt."""
        assert self.hypotheses is not None
        raise NotImplementedError

    def _geometric_reasoning(self) -> str:
        """Return the geometric reasoning prompt."""
        assert self.geometric is not None
        raise NotImplementedError

    def prompt(
        self,
        behavior: str,
        behavior_kwargs: Optional[Dict[str, Any]] = None,
        example: bool = False,
        openai: bool = True,
    ) -> Union[str, List[Dict[str, Any]]]:
        """Return the prompt for the behavior."""
        behaviors = {
            "goal_prediction": self._goal_prediction,
            "task_planning": self._task_planning,
            "hypothesis_generation": self._hypothesis_generation,
            "geometric_reasoning": self._geometric_reasoning,
        }

        if behavior in behaviors:
            behavior_kwargs = behavior_kwargs if behavior_kwargs is not None else {}
            query, response = behaviors[behavior](**behavior_kwargs)
            if example and response is None:
                raise ValueError(
                    f"Example response not provided for Behavior '{behavior}'."
                )
        else:
            raise ValueError(f"Behavior '{behavior}' is not supported.")

        if openai:
            assert self.role is not None
            if self.name_query is not None and self.name_response is not None:
                query_prompt = format_openai(
                    role=self.role, content=query, name=self.name_query
                )
                prompt = [query_prompt]
                if example:
                    response_prompt = format_openai(
                        role=self.role, content=response, name=self.name_response
                    )
                    prompt.append(response_prompt)
            else:
                prompt = format_openai(
                    role=self.role,
                    content=query + response if example else query,
                    name=self.name,
                )
                prompt = [prompt]
        else:
            prompt = query + response if example else query

        return prompt


class BehaviorPromptManager:

    def __init__(
        self,
        task_prompt: Dict[str, Any],
        system_prompts: Union[List[str], List[pathlib.Path]],
        example_prompts: Optional[Union[List[str], List[pathlib.Path]]] = None,
        event_prompts: Optional[Union[List[str], List[pathlib.Path]]] = None,
        openai: bool = True,
    ) -> None:
        """Initialize behavior prompt manager."""
        # Instantiate prompts.
        self._task_prompt = BehaviorPrompt.from_config(task_prompt)
        system_prompts = [SystemPrompt.from_yaml(prompt) for prompt in system_prompts]
        self._system_prompts: Dict[str, SystemPrompt] = {
            prompt.behavior: prompt for prompt in system_prompts
        }
        self._example_prompts: List[BehaviorPrompt] = (
            [BehaviorPrompt.from_yaml(prompt) for prompt in example_prompts]
            if example_prompts is not None
            else []
        )
        self._event_prompts: List[BehaviorPrompt] = (
            [BehaviorPrompt.from_yaml(prompt) for prompt in event_prompts]
            if event_prompts is not None
            else []
        )
        self._openai = openai

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BehaviorPromptManager":
        """Create a behavior prompt manager from a config."""
        return cls(**config)

    @classmethod
    def from_yaml(cls, file: Union[str, pathlib.Path]) -> "BehaviorPromptManager":
        """Create a behavior prompt manager from a YAML file."""
        if isinstance(file, str):
            file = pathlib.Path(file)

        with file.open("r") as f:
            config = yaml.safe_load(f)

        assert config["prompt"] == "BehaviorPromptManager"
        config = config["prompt_kwargs"]

        return cls.from_config(config)

    @property
    def task_prompt(self) -> BehaviorPrompt:
        """Get the task prompt."""
        return self._task_prompt

    @property
    def event_prompts(self) -> List[BehaviorPrompt]:
        """Get the event prompts."""
        return self._event_prompts

    @event_prompts.setter
    def event_prompts(self, event_prompts: List[BehaviorPrompt]) -> None:
        """Set the event prompts."""
        self._event_prompts = event_prompts

    @property
    def behaviors(self) -> Set[str]:
        """Return the behaviors."""
        return set(self._system_prompts.keys())

    def generate_prompt(
        self,
        behavior: str,
        use_examples: bool = True,
        use_events: bool = False,
        openai: Optional[bool] = None,
    ) -> Union[str, List[Dict[str, Any]]]:
        """Generate a prompt for the behavior."""
        openai = self._openai if openai is None else openai
        prompts = []

        # System prompt.
        if behavior in self._system_prompts:
            system_prompt = self._system_prompts[behavior]
            prompts.append(system_prompt.prompt(openai=openai))
        else:
            raise ValueError(f"Behavior '{behavior}' is not supported.")

        # Example prompts.
        if use_examples:
            for example_prompt in self._example_prompts:
                prompt = example_prompt.prompt(
                    behavior,
                    behavior_kwargs=system_prompt.behavior_kwargs,
                    example=True,
                    openai=openai,
                )
                prompts.append(prompt)

        # Event history prompts.
        if use_events:
            raise NotImplementedError

        # Current task prompt.
        task_prompt = self._task_prompt.prompt(
            behavior, behavior_kwargs=system_prompt.behavior_kwargs, openai=openai
        )
        prompts.append(task_prompt)

        if openai:
            temp = []
            for p in prompts:
                if isinstance(p, list):
                    temp.extend(p)
                else:
                    temp.append(p)
            prompts = temp
        else:
            prompts = "\n\n".join(prompts)

        return prompts


class Llama2Prompt:

    SYSTEM_PROMPT = """<s>[INST] <<SYS>>
{SYSTEM_MESSAGE}
<</SYS>>

"""

    def __init__(self, system_message: str):
        """Initialize Llama 2 prompt."""
        self._system_message = system_message
        self._messages = []
        self._responses = []

    def add_message(self, message: str):
        """Add a message to the prompt."""
        self._messages.append(message)

    def add_response(self, response: str):
        """Add a response to the prompt."""
        self._responses.append(response)

    def __str__(self) -> str:
        """Return the prompt as a string."""
        prompt = self.SYSTEM_PROMPT.replace("{SYSTEM_MESSAGE}", self._system_message)
        for i in range(len(self._messages + self._responses)):
            if i % 2 == 0:
                if i == 0:
                    prompt += f"{self._messages[i]} [/INST]"
                else:
                    prompt += f" {self._messages[i]} [/INST]"
            else:
                prompt += f" {self._responses[i]} </s><s>[INST]"

        return prompt
