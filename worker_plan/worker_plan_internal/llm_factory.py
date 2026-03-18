"""
Create a LLM instances.

PROMPT> python -m worker_plan_internal.llm_factory
"""
import logging
import os
from typing import Optional, Any
from worker_plan_api.planexe_dotenv import PlanExeDotEnv
from worker_plan_internal.utils.planexe_llmconfig import PlanExeLLMConfig
from worker_plan_api.model_profile import ModelProfileEnum, resolve_model_profile_from_env
from llama_index.core.llms.llm import LLM
# from llama_index.llms.mistralai import MistralAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openai import OpenAI
# from llama_index.llms.together import TogetherLLM
# from llama_index.llms.groq import Groq
from llama_index.llms.lmstudio import LMStudio
from llama_index.llms.openrouter import OpenRouter

# Claude Code OAuth tokens (sk-ant-oat*) require this beta header.
# Generated via: claude setup-token  (needs Claude Pro/Max subscription)
_CLAUDE_OAUTH_TOKEN_PREFIX = "sk-ant-oat"
_CLAUDE_OAUTH_BETA_HEADER = "oauth-2025-04-20"
from worker_plan_internal.llm_util.ollama_info import OllamaInfo
from worker_plan_internal.llm_util.thinking_aware_openai_like import ThinkingAwareOpenAILike
from worker_plan_api.llm_info import LLMConfigItem, LLMInfo, OllamaStatus

# You can disable this if you don't want to send app info to OpenRouter.
SEND_APP_INFO_TO_OPENROUTER = True

# This is a special case. It will cycle through the available LLM models, if the first one fails, try the next one.
SPECIAL_AUTO_ID = 'auto'
SPECIAL_AUTO_LABEL = 'Auto'

logger = logging.getLogger(__name__)

__all__ = ["get_llm", "LLMInfo", "get_llm_names_by_priority", "SPECIAL_AUTO_ID", "is_valid_llm_name", "obtain_llm_info"]


def _resolve_model_profile(model_profile: Optional[ModelProfileEnum | str]) -> ModelProfileEnum:
    if isinstance(model_profile, ModelProfileEnum):
        return model_profile
    if isinstance(model_profile, str):
        for enum_value in ModelProfileEnum:
            if enum_value.value == model_profile.strip().lower():
                return enum_value
    return resolve_model_profile_from_env()


def _load_llm_config(model_profile: Optional[ModelProfileEnum | str]) -> PlanExeLLMConfig:
    resolved_profile = _resolve_model_profile(model_profile)
    planexe_llmconfig = PlanExeLLMConfig.load(model_profile=resolved_profile)
    try:
        from worker_plan_internal.llm_util.model_pricing import load_pricing_from_llm_config
        load_pricing_from_llm_config(planexe_llmconfig.llm_config_dict)
    except Exception:
        logger.debug("Failed to load model pricing from config", exc_info=True)
    try:
        from worker_plan_internal.llm_util.usage_metrics import install_anthropic_usage_hook
        install_anthropic_usage_hook()
    except Exception:
        logger.debug("Failed to install Anthropic usage hook", exc_info=True)
    return planexe_llmconfig


def obtain_llm_info(model_profile: Optional[ModelProfileEnum | str] = None) -> LLMInfo:
    """
    Returns a list of available LLM names and Ollama status.
    """

    planexe_llmconfig = _load_llm_config(model_profile)

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

    # The rest are the LLM models specified in the selected llm_config/<profile>.json file.
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
            error_message = (
                f"Problem with config `\"{config_id}\"`: The model `\"{model}\"` is not available in Ollama. "
                "Compare model names in the selected `llm_config/<profile>.json` file with the names available in Ollama."
            )
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

def get_llm_names_by_priority(model_profile: Optional[ModelProfileEnum | str] = None) -> list[str]:
    """
    Returns a list of LLM names sorted by priority.
    Lowest values comes first.
    Highest values comes last.
    """
    planexe_llmconfig = _load_llm_config(model_profile)
    configs = [(name, config) for name, config in planexe_llmconfig.llm_config_dict.items() if config.get("priority") is not None]
    configs.sort(key=lambda x: x[1].get("priority", 0))
    return [name for name, _ in configs]

def is_valid_llm_name(llm_name: str, model_profile: Optional[ModelProfileEnum | str] = None) -> bool:
    """
    Returns True if the LLM name is valid, False otherwise.
    """
    planexe_llmconfig = _load_llm_config(model_profile)
    return llm_name in planexe_llmconfig.llm_config_dict

def get_llm(llm_name: Optional[str] = None, model_profile: Optional[ModelProfileEnum | str] = None, **kwargs: Any) -> LLM:
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

    planexe_llmconfig = _load_llm_config(model_profile)

    if llm_name == SPECIAL_AUTO_ID:
        logger.error(f"The special {SPECIAL_AUTO_ID!r} is not a LLM model that can be created. Please use a valid LLM name.")
        raise ValueError(f"The special {SPECIAL_AUTO_ID!r} is not a LLM model that can be created. Please use a valid LLM name.")

    if not is_valid_llm_name(llm_name, model_profile=model_profile):
        logger.error(
            f"Cannot create LLM, the llm_name {llm_name!r} is not found in the selected llm_config/<profile>.json."
        )
        raise ValueError(
            f"Cannot create LLM, the llm_name {llm_name!r} is not found in the selected llm_config/<profile>.json."
        )

    config = planexe_llmconfig.llm_config_dict[llm_name]
    class_name = config.get("class")
    arguments = config.get("arguments", {})

    # Override with any kwargs passed to get_llm()
    arguments.update(kwargs)

    _claude_oauth_token: Optional[str] = None  # set below if OAuth detected
    if class_name == "Anthropic":
        # Auto-detect Claude Code OAuth tokens.
        # Regular API keys (sk-ant-api*): sent via x-api-key — no changes needed.
        # OAuth tokens (sk-ant-oat*) from `claude setup-token`:
        #   - Must use Authorization: Bearer (x-api-key causes 401)
        #   - Require anthropic-beta: oauth-2025-04-20 header
        #   - llama_index always passes api_key to the underlying SDK client, so
        #     we can't suppress x-api-key at the argument level. Instead, after
        #     llama_index instantiates the LLM we replace _client/_aclient with
        #     SDK clients built from auth_token= (Bearer) instead of api_key=.
        api_key = arguments.get("api_key", "")
        if api_key and api_key.startswith(_CLAUDE_OAUTH_TOKEN_PREFIX):
            _claude_oauth_token = api_key
            logger.debug("Claude Code OAuth token detected — will apply Bearer auth post-init.")

    if class_name == "OpenRouter" and SEND_APP_INFO_TO_OPENROUTER:
        # https://openrouter.ai/rankings
        # https://openrouter.ai/docs/api-reference/overview#headers
        arguments_extra = {
            "additional_kwargs": {
                "extra_headers": {
                    "HTTP-Referer": "https://github.com/PlanExeOrg/PlanExe",
                    "X-Title": "PlanExe - the premier planning tool for AI agents"
                }
            }
        }
        arguments.update(arguments_extra)

    # Dynamically instantiate the class
    try:
        llm_class = globals()[class_name]  # Get class from global scope
        llm_instance = llm_class(**arguments)

        # Post-init: swap underlying SDK clients for OAuth tokens.
        # llama_index always builds its internal client with api_key= (x-api-key header),
        # but Anthropic rejects requests that send x-api-key alongside a Bearer token.
        # Replacing _client/_aclient with auth_token=-based SDK clients fixes this.
        #
        # Additionally, when using beta.messages.parse(), the SDK's structured-outputs
        # beta (structured-outputs-2025-12-15) overwrites default_headers["anthropic-beta"],
        # causing the OAuth beta to be lost. We use an httpx event hook at the request level
        # to ensure the OAuth beta is always present, regardless of what the SDK adds.
        if _claude_oauth_token and class_name == "Anthropic":
            import anthropic as _anthropic_sdk
            import httpx
            
            # Event hook: ensures OAuth beta is always present in anthropic-beta header
            def _add_oauth_beta_header(request: httpx.Request) -> None:
                """Append OAuth beta to anthropic-beta header at the HTTP request level."""
                existing = request.headers.get("anthropic-beta", "")
                if _CLAUDE_OAUTH_BETA_HEADER not in existing:
                    # Append OAuth beta to any existing anthropic-beta value (e.g., from parse())
                    if existing:
                        new_value = f"{_CLAUDE_OAUTH_BETA_HEADER},{existing}"
                    else:
                        new_value = _CLAUDE_OAUTH_BETA_HEADER
                    request.headers["anthropic-beta"] = new_value
            
            # Temporarily remove ANTHROPIC_API_KEY from env so the SDK doesn't
            # inject X-Api-Key alongside the Bearer token — the server rejects
            # requests that include both auth headers simultaneously.
            _saved_api_key = os.environ.pop("ANTHROPIC_API_KEY", None)
            try:
                # Build sync HTTP client with OAuth beta event hook
                sync_http_client = httpx.Client(
                    event_hooks={"request": [_add_oauth_beta_header]}
                )
                sync_client = _anthropic_sdk.Anthropic(
                    auth_token=_claude_oauth_token,
                    base_url=arguments.get("base_url"),
                    timeout=arguments.get("timeout", 60.0),
                    max_retries=arguments.get("max_retries", 3),
                    http_client=sync_http_client,
                )
                
                # Build async HTTP client with OAuth beta event hook
                async_http_client = httpx.AsyncClient(
                    event_hooks={"request": [_add_oauth_beta_header]}
                )
                async_client = _anthropic_sdk.AsyncAnthropic(
                    auth_token=_claude_oauth_token,
                    base_url=arguments.get("base_url"),
                    timeout=arguments.get("timeout", 60.0),
                    max_retries=arguments.get("max_retries", 3),
                    http_client=async_http_client,
                )
                
                llm_instance._client = sync_client
                llm_instance._aclient = async_client
            finally:
                if _saved_api_key:
                    os.environ["ANTHROPIC_API_KEY"] = _saved_api_key
            logger.debug(
                "Replaced Anthropic SDK clients with Bearer auth (OAuth token) and "
                "httpx event hook to inject OAuth beta header at request level."
            )

        return llm_instance
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
