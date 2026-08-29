import unittest

import mcp_cloud.app as cloud_app


def _tool_desc(tool_defs, name: str) -> str:
    for definition in tool_defs:
        if definition.name == name:
            return definition.description
    raise AssertionError(f"Tool not found: {name}")


def _tool_def(tool_defs, name: str):
    for definition in tool_defs:
        if definition.name == name:
            return definition
    raise AssertionError(f"Tool not found: {name}")


class TestAllToolsHaveOutputSchema(unittest.TestCase):
    """Every tool must declare an output_schema so callers know the response shape."""

    def test_cloud_all_tools_have_output_schema(self):
        for definition in cloud_app.TOOL_DEFINITIONS:
            with self.subTest(tool=definition.name):
                self.assertIsNotNone(
                    definition.output_schema,
                    f"Cloud tool {definition.name!r} is missing output_schema",
                )


class TestPlanCreateInputSchemaHasUserApiKey(unittest.TestCase):
    """user_api_key must be in the visible plan_create input schema."""

    def test_cloud_plan_create_schema_has_user_api_key(self):
        props = cloud_app.PLAN_CREATE_INPUT_SCHEMA.get("properties", {})
        self.assertIn("user_api_key", props)


class TestPlanListInputSchemaHasUserApiKey(unittest.TestCase):
    """user_api_key must be in plan_list input schema but NOT required."""

    def test_cloud_plan_list_schema_has_optional_user_api_key(self):
        props = cloud_app.PLAN_LIST_INPUT_SCHEMA.get("properties", {})
        self.assertIn("user_api_key", props)
        required = cloud_app.PLAN_LIST_INPUT_SCHEMA.get("required", [])
        self.assertNotIn("user_api_key", required)


class TestPlanRetryInputSchemaDefaults(unittest.TestCase):
    """plan_retry should default model_profile to baseline."""

    def test_cloud_plan_retry_schema_defaults_model_profile(self):
        props = cloud_app.PLAN_RETRY_INPUT_SCHEMA.get("properties", {})
        model_profile = props.get("model_profile", {})
        self.assertEqual(model_profile.get("default"), "baseline")


class TestPlanResumeInputSchemaDefaults(unittest.TestCase):
    """plan_resume should default model_profile to baseline."""

    def test_cloud_plan_resume_schema_defaults_model_profile(self):
        props = cloud_app.PLAN_RESUME_INPUT_SCHEMA.get("properties", {})
        model_profile = props.get("model_profile", {})
        self.assertEqual(model_profile.get("default"), "baseline")


class TestExamplePromptsAnnotations(unittest.TestCase):
    def test_cloud_example_prompts_annotations(self):
        definition = _tool_def(cloud_app.TOOL_DEFINITIONS, "example_prompts")
        annotations = definition.annotations or {}
        self.assertTrue(annotations.get("readOnlyHint"))
        self.assertFalse(annotations.get("destructiveHint"))
        self.assertTrue(annotations.get("idempotentHint"))
        self.assertFalse(annotations.get("openWorldHint"))


class TestModelProfilesAnnotations(unittest.TestCase):
    def test_cloud_model_profiles_annotations(self):
        definition = _tool_def(cloud_app.TOOL_DEFINITIONS, "model_profiles")
        annotations = definition.annotations or {}
        self.assertTrue(annotations.get("readOnlyHint"))
        self.assertFalse(annotations.get("destructiveHint"))
        self.assertTrue(annotations.get("idempotentHint"))
        self.assertFalse(annotations.get("openWorldHint"))


class TestRemainingToolAnnotations(unittest.TestCase):
    def test_cloud_remaining_tool_annotations(self):
        expected = {
            "plan_create": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
            "plan_status": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
            "plan_stop": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": False},
            "plan_retry": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
            "plan_resume": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
            "plan_file_info": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
            "plan_list": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
            "send_feedback": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
        }
        for tool_name, expected_annotations in expected.items():
            with self.subTest(tool=tool_name):
                definition = _tool_def(cloud_app.TOOL_DEFINITIONS, tool_name)
                self.assertEqual(definition.annotations, expected_annotations)


class TestExamplePlansAnnotations(unittest.TestCase):
    def test_cloud_example_plans_annotations(self):
        definition = _tool_def(cloud_app.TOOL_DEFINITIONS, "example_plans")
        annotations = definition.annotations or {}
        self.assertTrue(annotations.get("readOnlyHint"))
        self.assertFalse(annotations.get("destructiveHint"))
        self.assertTrue(annotations.get("idempotentHint"))
        self.assertFalse(annotations.get("openWorldHint"))


class TestCloudToolSurfaceConsistency(unittest.TestCase):
    def test_cloud_exposes_example_plans_tool(self):
        cloud_tool_names = {definition.name for definition in cloud_app.TOOL_DEFINITIONS}
        self.assertIn("example_plans", cloud_tool_names)

    def test_cloud_exposes_model_profiles_tool(self):
        cloud_tool_names = {definition.name for definition in cloud_app.TOOL_DEFINITIONS}
        self.assertIn("model_profiles", cloud_tool_names)

    def test_cloud_exposes_plan_retry_tool(self):
        cloud_tool_names = {definition.name for definition in cloud_app.TOOL_DEFINITIONS}
        self.assertIn("plan_retry", cloud_tool_names)

    def test_cloud_exposes_plan_resume_tool(self):
        cloud_tool_names = {definition.name for definition in cloud_app.TOOL_DEFINITIONS}
        self.assertIn("plan_resume", cloud_tool_names)

    def test_cloud_exposes_plan_file_info(self):
        cloud_tool_names = {definition.name for definition in cloud_app.TOOL_DEFINITIONS}
        self.assertIn("plan_file_info", cloud_tool_names)

    def test_cloud_exposes_plan_list_tool(self):
        cloud_tool_names = {definition.name for definition in cloud_app.TOOL_DEFINITIONS}
        self.assertIn("plan_list", cloud_tool_names)

    def test_cloud_exposes_send_feedback_tool(self):
        cloud_tool_names = {definition.name for definition in cloud_app.TOOL_DEFINITIONS}
        self.assertIn("send_feedback", cloud_tool_names)

    def test_cloud_instructions_reference_send_feedback(self):
        self.assertIn("send_feedback", cloud_app.PLANEXE_SERVER_INSTRUCTIONS)

    def test_cloud_instructions_reference_download_tool(self):
        self.assertIn("plan_file_info", cloud_app.PLANEXE_SERVER_INSTRUCTIONS)

    def test_cloud_plan_create_description_references_download_tool(self):
        description = _tool_desc(cloud_app.TOOL_DEFINITIONS, "plan_create")
        self.assertIn("plan_file_info", description)

    def test_cloud_instructions_include_plan_status_state_contract(self):
        instructions = cloud_app.PLANEXE_SERVER_INSTRUCTIONS
        self.assertIn("pending/processing", instructions)
        self.assertIn("completed", instructions)
        self.assertIn("failed", instructions)
        self.assertNotIn("running/stopping", instructions)
        self.assertIn("pending for longer than 5 minutes", instructions)
        self.assertIn("longer than 20 minutes", instructions)
        self.assertIn("PlanExeOrg/PlanExe/issues", instructions)

    def test_cloud_plan_status_description_includes_state_contract(self):
        description = _tool_desc(cloud_app.TOOL_DEFINITIONS, "plan_status")
        self.assertIn("pending/processing", description)
        self.assertIn("completed", description)
        self.assertIn("failed", description)
        self.assertNotIn("running/stopping", description)
        self.assertIn("pending for >5 minutes", description)
        self.assertIn(">20 minutes", description)
        self.assertIn("PlanExeOrg/PlanExe/issues", description)

    def test_cloud_instructions_include_model_profiles_unavailable_guidance(self):
        instructions = cloud_app.PLANEXE_SERVER_INSTRUCTIONS
        self.assertIn("MODEL_PROFILES_UNAVAILABLE", instructions)

    def test_cloud_prompt_schema_includes_prompt_shape_guidance(self):
        prompt_schema = cloud_app.PLAN_CREATE_INPUT_SCHEMA["properties"]["prompt"]["description"]
        self.assertIn("300-800 words", prompt_schema)
        self.assertIn("objective, scope, constraints, timeline, stakeholders, budget/resources, and success criteria", prompt_schema)


class TestMCPServerCanonicalOutputSchema(unittest.TestCase):
    """Registered tools must advertise the canonical outputSchema from TOOL_DEFINITIONS."""

    def test_flat_tools_use_canonical_output_schema(self):
        """Flat-schema tools must have their canonical outputSchema injected."""
        from mcp_cloud.http_server import mcp_server

        for tool_def in cloud_app.TOOL_DEFINITIONS:
            if tool_def.output_schema is None:
                continue
            if "oneOf" in tool_def.output_schema:
                continue  # oneOf schemas are tested separately
            with self.subTest(tool=tool_def.name):
                registered_tool = mcp_server._tool_manager.get_tool(tool_def.name)
                self.assertIsNotNone(
                    registered_tool,
                    f"MCP tool {tool_def.name!r} not registered",
                )
                self.assertEqual(
                    registered_tool.output_schema,
                    tool_def.output_schema,
                    f"MCP tool {tool_def.name!r} outputSchema does not match TOOL_DEFINITIONS",
                )

    def test_oneof_tools_have_no_output_schema(self):
        """oneOf schemas must NOT be advertised — MCP clients reject them."""
        from mcp_cloud.http_server import mcp_server

        oneof_tools = [
            td.name for td in cloud_app.TOOL_DEFINITIONS
            if td.output_schema and "oneOf" in td.output_schema
        ]
        self.assertTrue(len(oneof_tools) > 0, "Expected at least one oneOf tool")
        for name in oneof_tools:
            with self.subTest(tool=name):
                registered_tool = mcp_server._tool_manager.get_tool(name)
                self.assertIsNone(
                    registered_tool.output_schema,
                    f"MCP tool {name!r} must not advertise oneOf outputSchema",
                )

    def test_all_tool_definitions_registered_in_mcp_server(self):
        """Every tool in TOOL_DEFINITIONS must be registered in the MCP server."""
        from mcp_cloud.http_server import mcp_server

        for tool_def in cloud_app.TOOL_DEFINITIONS:
            with self.subTest(tool=tool_def.name):
                registered_tool = mcp_server._tool_manager.get_tool(tool_def.name)
                self.assertIsNotNone(
                    registered_tool,
                    f"TOOL_DEFINITIONS has {tool_def.name!r} but the MCP server does not",
                )

    def test_plan_file_info_canonical_schema_has_three_oneof_variants(self):
        """plan_file_info canonical schema must have oneOf with error, not-ready, and ready shapes."""
        tool_def = _tool_def(cloud_app.TOOL_DEFINITIONS, "plan_file_info")
        schema = tool_def.output_schema
        self.assertIn("oneOf", schema)
        self.assertEqual(len(schema["oneOf"]), 3)
        error_variant = schema["oneOf"][0]
        self.assertIn("error", error_variant.get("required", []))

    def test_plan_status_canonical_schema_has_two_oneof_variants(self):
        """plan_status canonical schema must have oneOf with error and success shapes."""
        tool_def = _tool_def(cloud_app.TOOL_DEFINITIONS, "plan_status")
        schema = tool_def.output_schema
        self.assertIn("oneOf", schema)
        self.assertEqual(len(schema["oneOf"]), 2)
        error_variant = schema["oneOf"][0]
        self.assertIn("error", error_variant.get("required", []))

    def test_simple_tools_have_flat_schema(self):
        """Tools with a single success shape should not have oneOf."""
        simple_tools = ["example_plans", "example_prompts", "model_profiles",
                        "plan_create", "plan_resume", "plan_list", "send_feedback"]
        for name in simple_tools:
            with self.subTest(tool=name):
                tool_def = _tool_def(cloud_app.TOOL_DEFINITIONS, name)
                schema = tool_def.output_schema
                self.assertNotIn(
                    "oneOf", schema,
                    f"{name} should have a flat schema, not oneOf",
                )

    def test_tool_functions_return_plain_call_tool_result(self):
        """No tool function should use Annotated return type (regression guard)."""
        import typing
        from mcp_cloud import http_server

        tool_funcs = [
            http_server.example_plans,
            http_server.example_prompts,
            http_server.model_profiles,
            http_server.plan_create,
            http_server.plan_status,
            http_server.plan_stop,
            http_server.plan_retry,
            http_server.plan_resume,
            http_server.plan_file_info,
            http_server.plan_list,
            http_server.send_feedback,
        ]
        for func in tool_funcs:
            with self.subTest(func=func.__name__):
                hints = typing.get_type_hints(func, include_extras=True)
                ret = hints.get("return")
                origin = getattr(ret, "__class__", None)
                self.assertNotEqual(
                    getattr(origin, "__name__", ""),
                    "_AnnotatedAlias",
                    f"{func.__name__} return type must be plain CallToolResult, "
                    f"not Annotated[CallToolResult, ...]",
                )

    def test_plan_file_info_not_derived_from_pydantic(self):
        """plan_file_info must not have a schema derived from PlanFileInfoOutput."""
        from mcp_cloud.http_server import mcp_server
        from mcp_cloud.tool_models import PlanFileInfoOutput

        registered_tool = mcp_server._tool_manager.get_tool("plan_file_info")
        pydantic_schema = PlanFileInfoOutput.model_json_schema()
        # oneOf schemas are not advertised, so output_schema should be None.
        # Either way, it must NOT equal the flat Pydantic derivation.
        self.assertNotEqual(
            registered_tool.output_schema,
            pydantic_schema,
            "plan_file_info outputSchema looks like it was derived from "
            "PlanFileInfoOutput instead of using the canonical schema",
        )


if __name__ == "__main__":
    unittest.main()
