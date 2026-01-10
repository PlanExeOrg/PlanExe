"""
Load PlanExe's llm_config.json file, containing LLM configurations

PROMPT> python -m worker_plan_internal.utils.planexe_llmconfig
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json
from worker_plan_api.planexe_config import PlanExeConfig
from worker_plan_api.planexe_dotenv import PlanExeDotEnv
import logging

logger = logging.getLogger(__name__)

@dataclass
class PlanExeLLMConfig:
    llm_config_json_path: Path
    llm_config_dict_raw: dict[str, Any]
    llm_config_dict: dict[str, Any]

    @classmethod
    def load(cls):
        config = PlanExeConfig.load()
        config.raise_if_required_files_not_found()
        planexe_dotenv = PlanExeDotEnv.load()

        llm_config_json_path = config.llm_config_json_path
        llm_config_dict_raw = cls.load_llm_config(llm_config_json_path)
        llm_config_dict = cls.substitute_env_vars(llm_config_dict_raw, planexe_dotenv.dotenv_dict)

        return cls(
            llm_config_json_path=llm_config_json_path,
            llm_config_dict_raw=llm_config_dict_raw,
            llm_config_dict=llm_config_dict
        )

    @classmethod
    def load_llm_config(cls, llm_config_json_path: Path) -> Dict[str, Any]:
        """Loads the configuration from a JSON file."""
        try:
            with open(llm_config_json_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Warning: llm_config.json not found at {llm_config_json_path}. Using an empty dictionary.")
            return {}
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {llm_config_json_path}: {e}")

    @classmethod
    def substitute_env_vars(cls, config: Dict[str, Any], env_vars: Dict[str, str]) -> Dict[str, Any]:
        """Recursively substitutes environment variables in the configuration."""

        def replace_value(value: Any) -> Any:
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]  # Extract variable name
                if var_name in env_vars:
                    return env_vars[var_name]
                else:
                    logger.warning(f"Warning: Environment variable '{var_name}' not found.")
                    return value  # Or raise an error if you prefer strict enforcement
            return value

        def process_item(item):
            if isinstance(item, dict):
                return {k: process_item(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [process_item(i) for i in item]
            else:
                return replace_value(item)

        return process_item(config)

    def __repr__(self):
        return f"PlanExeLLMConfig(llm_config_json_path={self.llm_config_json_path!r}, llm_config_dict.keys()={self.llm_config_dict.keys()!r})"

if __name__ == "__main__":
    llm_config = PlanExeLLMConfig.load()
    print(llm_config)    
    print(f"\nllm_config.llm_config_dict_raw: {llm_config.llm_config_dict_raw!r}")
    # print(f"\nllm_config.llm_config_dict: {llm_config.llm_config_dict!r}")
