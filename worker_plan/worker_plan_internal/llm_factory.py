"""
Create a LLM instances.

PROMPT> python -m worker_plan_internal.llm_factory
"""
import logging
from typing import Optional, Any
from worker_plan_api.planexe_dotenv import PlanExeDotEnv
from worker_plan_api.planexe_config import PlanExeConfig, PlanExeConfigError
from worker_plan_internal.utils.planexe_llmconfig import PlanExeLLMConfig
from llama_index.core.llms.llm import LLM
# from llama_index.llms.mistralai import MistralAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openai import OpenAI
# from llama_index.llms.together import TogetherLLM
# from llama_index.llms.groq import Groq
from llama_index.llms.lmstudio import LMStudio
from llama_index.llms.openrouter import OpenRouter
from worker_plan_internal.llm_util.ollama_info import OllamaInfo
from worker_plan_api.llm_info import LLMConfigItem, LLMInfo, OllamaStatus

# You can disable this if you don't want to send app info to OpenRouter.
SEND_APP_INFO_TO_OPENROUTER = True

# This is a special case. It will cycle through the available LLM models, if the first one fails, try the next one.
SPECIAL_AUTO_ID = 'auto'
SPECIAL_AUTO_LABEL = 'Auto'

logger = logging.getLogger(__name__)

__all__ = ["get_llm", "LLMInfo", "get_llm_names_by_priority", "SPECIAL_AUTO_ID", "is_valid_llm_name", "obtain_llm_info"]

planexe_llmconfig = PlanExeLLMConfig.load()

def obtain_llm_info() -> LLMInfo:
    """
    Returns a list of available LLM names and Ollama status.
    """

    # Probe each Ollama service endpoint just once.
    error_message_list = []
    ollama_info_per_host = {}
    count_running = 0
    count_not_running = 0
    for config_id, config in planexe_llmconfig.llm_config_dict.items():
        if config.get("class") != "Ollama":
            continue
        arguments = config.get("arguments", {})
        model = arguments.get("model", None)
        base_url = arguments.get("base_url", None)

        if base_url in ollama_info_per_host:
            # Already got info for this host. No need to get it again.
            continue

        ollama_info = OllamaInfo.obtain_info(base_url=base_url)
        ollama_info_per_host[base_url] = ollama_info

        running_on = "localhost" if base_url is None else base_url

        if ollama_info.is_running:
            count_running += 1
        else:
            count_not_running += 1

        if not ollama_info.is_running:
            print(f"Ollama is not running on {running_on}. Please start the Ollama service, in order to use the models via Ollama.")
        elif ollama_info.error_message:
            print(f"Error message: {ollama_info.error_message}")
            error_message_list.append(ollama_info.error_message)

    # Prepare the list of available LLM config items.
    llm_config_items = []

    # This is a special case. It will cycle through the available LLM models, if the first one fails, try the next one.
    llm_config_items.append(LLMConfigItem(id=SPECIAL_AUTO_ID, label=SPECIAL_AUTO_LABEL))

    # The rest are the LLM models specified in the llm_config.json file.
    for config_id, config in planexe_llmconfig.llm_config_dict.items():
        priority = config.get("priority", None)
        if priority:
            label_with_priority = f"{config_id} (prio: {priority})"
        else:
            label_with_priority = config_id

        if config.get("class") != "Ollama":
            item = LLMConfigItem(id=config_id, label=label_with_priority)
            llm_config_items.append(item)
            continue

        # Get info about the each LLM config item that is using Ollama.
        arguments = config.get("arguments", {})
        model = arguments.get("model", None)
        base_url = arguments.get("base_url", None)

        ollama_info = ollama_info_per_host[base_url]

        is_model_available = ollama_info.is_model_available(model)
        if is_model_available:
            label = label_with_priority
        else:
            label = f"{label_with_priority} ❌ unavailable"
        
        if ollama_info.is_running and not is_model_available:
            error_message = f"Problem with config `\"{config_id}\"`: The model `\"{model}\"` is not available in Ollama. Compare model names in `llm_config.json` with the names available in Ollama."
            error_message_list.append(error_message)
        
        item = LLMConfigItem(id=config_id, label=label)
        llm_config_items.append(item)

    if count_not_running == 0 and count_running > 0:
        ollama_status = OllamaStatus.ollama_running
    elif count_not_running > 0 and count_running == 0:
        ollama_status = OllamaStatus.ollama_not_running
    elif count_not_running > 0 and count_running > 0:
        ollama_status = OllamaStatus.mixed
    else:
        ollama_status = OllamaStatus.no_ollama_models

    return LLMInfo(
        llm_config_items=llm_config_items, 
        ollama_status=ollama_status,
        error_message_list=error_message_list,
    )

def get_llm_names_by_priority() -> list[str]:
    """
    Returns a list of LLM names sorted by priority.
    Lowest values comes first.
    Highest values comes last.
    """
    configs = [(name, config) for name, config in planexe_llmconfig.llm_config_dict.items() if config.get("priority") is not None]
    configs.sort(key=lambda x: x[1].get("priority", 0))
    return [name for name, _ in configs]

def is_valid_llm_name(llm_name: str) -> bool:
    """
    Returns True if the LLM name is valid, False otherwise.
    """
    return llm_name in planexe_llmconfig.llm_config_dict

def get_llm(llm_name: Optional[str] = None, **kwargs: Any) -> LLM:
    """
    Returns an LLM instance based on the config.json file or a fallback default.

    :param llm_name: The name/key of the LLM to instantiate.
                     If None, falls back to DEFAULT_LLM from the environment/.env (or 'ollama-llama3.1').
    :param kwargs: Additional keyword arguments to override default model parameters.
    :return: An instance of a LlamaIndex LLM class.
    """
    if not llm_name:
        planexe_dotenv = PlanExeDotEnv.load()
        llm_name = planexe_dotenv.get("DEFAULT_LLM", "ollama-llama3.1")

    if llm_name == SPECIAL_AUTO_ID:
        logger.error(f"The special {SPECIAL_AUTO_ID!r} is not a LLM model that can be created. Please use a valid LLM name.")
        raise ValueError(f"The special {SPECIAL_AUTO_ID!r} is not a LLM model that can be created. Please use a valid LLM name.")

    if not is_valid_llm_name(llm_name):
        logger.error(f"Cannot create LLM, the llm_name {llm_name!r} is not found in llm_config.json.")
        raise ValueError(f"Cannot create LLM, the llm_name {llm_name!r} is not found in llm_config.json.")

    config = planexe_llmconfig.llm_config_dict[llm_name]
    class_name = config.get("class")
    arguments = config.get("arguments", {})

    # Override with any kwargs passed to get_llm()
    arguments.update(kwargs)

    if class_name == "OpenRouter" and SEND_APP_INFO_TO_OPENROUTER:
        # https://openrouter.ai/rankings
        # https://openrouter.ai/docs/api-reference/overview#headers
        arguments_extra = {
            "additional_kwargs": {
                "extra_headers": {
                    "HTTP-Referer": "https://github.com/neoneye/PlanExe",
                    "X-Title": "PlanExe"
                }
            }
        }
        arguments.update(arguments_extra)

    # Dynamically instantiate the class
    try:
        llm_class = globals()[class_name]  # Get class from global scope
        return llm_class(**arguments)
    except KeyError:
        raise ValueError(f"Invalid LLM class name in config.json: {class_name}")
    except TypeError as e:
        raise ValueError(f"Error instantiating {class_name} with arguments: {e}")

if __name__ == '__main__':
    llm_names = get_llm_names_by_priority()
    print("LLM names by priority:")
    for llm_name in llm_names:
        print(f"- {llm_name}")
    print("\n\nTesting the LLMs:")
    try:
        llm = get_llm(llm_name="ollama-llama3.1")
        print(f"Successfully loaded LLM: {llm.__class__.__name__}")
        print(llm.complete("Hello, how are you?"))
    except ValueError as e:
        print(f"Error: {e}")
