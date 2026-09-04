import unittest
from unittest.mock import patch

from worker_plan_internal.llm_factory import get_llm
from worker_plan_internal.utils.planexe_llmconfig import PlanExeLLMConfig


def make_config(llm_config_dict: dict) -> PlanExeLLMConfig:
    return PlanExeLLMConfig(
        llm_config_json_path=None,
        llm_config_dict_raw=llm_config_dict,
        llm_config_dict=llm_config_dict,
    )


class TestGetLLMOpenRouterAdditionalKwargs(unittest.TestCase):
    def test_extra_body_from_config_survives_app_info_headers(self):
        provider_pin = {"order": ["baidu/fp8"], "allow_fallbacks": False}
        config = make_config({
            "pinned": {
                "class": "OpenRouter",
                "arguments": {
                    "model": "deepseek/deepseek-v4-flash-0731",
                    "api_key": "sk-test",
                    "additional_kwargs": {
                        "extra_body": {"provider": provider_pin}
                    },
                },
            }
        })

        with patch("worker_plan_internal.llm_factory._load_llm_config", return_value=config):
            llm = get_llm("pinned")

        self.assertEqual(llm.additional_kwargs["extra_body"], {"provider": provider_pin})
        self.assertIn("HTTP-Referer", llm.additional_kwargs["extra_headers"])


if __name__ == "__main__":
    unittest.main()
