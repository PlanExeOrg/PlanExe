import json
import unittest
import tempfile
import importlib.util
from pathlib import Path
from worker_plan_internal.llm_util.llm_executor import LLMExecutor, LLMModelBase, LLMModelWithInstance, PipelineStopRequested, ShouldStopCallbackParameters
from worker_plan_internal.llm_util.response_mockllm import ResponseMockLLM
from worker_plan_internal.llm_util.usage_metrics import set_usage_metrics_path
from llama_index.core.llms.llm import LLM

class TestLLMExecutor(unittest.TestCase):
    def test_simple(self):
        # Arrange
        llm = ResponseMockLLM(
            responses=["Hello, world!"],
        )
        llm_model = LLMModelWithInstance(llm)
        executor = LLMExecutor(llm_models=[llm_model])

        def execute_function(llm: LLM) -> str:
            return llm.complete("Hi").text

        # Act
        result = executor.run(execute_function)

        # Assert
        self.assertEqual(result, "Hello, world!")
        self.assertEqual(executor.attempt_count, 1)

    def test_fallback_to_the_2nd_llm(self):
        """Create two LLMs: one that fails, one that succeeds"""
        # Arrange
        bad_llm = ResponseMockLLM(responses=["raise:BAD"])
        good_llm = ResponseMockLLM(responses=["I'm the 2nd LLM"])
        llm_models = LLMModelWithInstance.from_instances([bad_llm, good_llm])
        executor = LLMExecutor(llm_models=llm_models)

        def execute_function(llm: LLM) -> str:
            return llm.complete("Hi").text

        # Act
        result = executor.run(execute_function)

        # Assert - should succeed with the good LLM after the bad one fails
        self.assertEqual(result, "I'm the 2nd LLM")
        self.assertEqual(executor.attempt_count, 2)
        self.assertFalse(executor.attempts[0].success)
        self.assertTrue(executor.attempts[1].success)

    def test_exhaust_all_llms_but_none_succeeds(self):
        """Create two LLMs that raise exceptions"""
        # Arrange
        bad1_llm = ResponseMockLLM(responses=["raise:BAD1"])
        bad2_llm = ResponseMockLLM(responses=["raise:BAD2"])
        llm_models = LLMModelWithInstance.from_instances([bad1_llm, bad2_llm])
        executor = LLMExecutor(llm_models=llm_models)

        def execute_function(llm: LLM) -> str:
            return llm.complete("Hi").text

        # Act
        with self.assertRaises(Exception) as context:
            executor.run(execute_function)

        # Assert
        self.assertIn("Failed to run. Exhausted all LLMs.", str(context.exception))
        self.assertEqual(executor.attempt_count, 2)
        self.assertIn("BAD1", str(context.exception))
        self.assertIn("BAD2", str(context.exception))
        self.assertFalse(executor.attempts[0].success)
        self.assertFalse(executor.attempts[1].success)

    def test_failure_inside_create_llm(self):
        """Simulate that the LLM cannot be created, due to a possible configuration issue."""
        # Arrange
        class BadLLMModel(LLMModelBase):
            def create_llm(self) -> LLM:
                raise ValueError("Cannot initialize this model")
            def __repr__(self) -> str:
                return "BadLLMModel()"
           
        bad_llm_model = BadLLMModel()
        executor = LLMExecutor(llm_models=[bad_llm_model])

        def execute_function(llm: LLM) -> str:
            return llm.complete("Hi").text

        # Act
        with self.assertRaises(Exception) as context:
            executor.run(execute_function)

        # Assert
        self.assertIn("Failed to run. Exhausted all LLMs.", str(context.exception))
        self.assertEqual(executor.attempt_count, 1)
        attempt0 = executor.attempts[0]
        self.assertIs(attempt0.llm_model, bad_llm_model)
        self.assertEqual(attempt0.stage, 'create')
        self.assertFalse(attempt0.success)
        self.assertIsNone(attempt0.result)
        self.assertIsInstance(attempt0.exception, ValueError)
        self.assertEqual(str(attempt0.exception), "Cannot initialize this model")

    def test_continue_execution_when_callback_does_not_raise(self):
        # Arrange
        llm0 = ResponseMockLLM(
            responses=["raise:BAD0"],
        )
        llm1 = ResponseMockLLM(
            responses=["I'm the last LLM"],
        )
        llm_models = LLMModelWithInstance.from_instances([llm0, llm1])

        def should_stop_callback(parameters: ShouldStopCallbackParameters) -> None:
            # Not raising means continue execution
            pass
        
        executor = LLMExecutor(llm_models=llm_models, should_stop_callback=should_stop_callback)

        def execute_function(llm: LLM) -> str:
            return llm.complete("Hi").text

        # Act
        result = executor.run(execute_function)

        # Assert
        self.assertEqual(result, "I'm the last LLM")
        self.assertEqual(executor.attempt_count, 2)
        self.assertFalse(executor.attempts[0].success)
        self.assertTrue(executor.attempts[1].success)

    def test_stop_execution_when_callback_raises_pipeline_stop_requested_with_one_llm(self):
        """Run the first LLM, and stop execution before the second LLM is run."""
        # Arrange
        llm0 = ResponseMockLLM(
            responses=["I'm the first LLM and I'm good"],
        )
        llm1 = ResponseMockLLM(
            responses=["I'm the last LLM and I'm never supposed to be run"],
        )
        llm_models = LLMModelWithInstance.from_instances([llm0, llm1])

        def should_stop_callback(parameters: ShouldStopCallbackParameters) -> None:
            # Stop execution by raising PipelineStopRequested
            raise PipelineStopRequested("Stopping execution after first successful attempt")
        
        executor = LLMExecutor(llm_models=llm_models, should_stop_callback=should_stop_callback)

        def execute_function(llm: LLM) -> str:
            return llm.complete("Hi").text

        # Act
        with self.assertRaises(PipelineStopRequested) as context:
            executor.run(execute_function)

        # Assert
        self.assertIn("Stopping execution after first successful attempt", str(context.exception))
        self.assertEqual(executor.attempt_count, 1)
        attempt0 = executor.attempts[0]
        self.assertTrue(attempt0.success)
        self.assertEqual(attempt0.result, "I'm the first LLM and I'm good")

    def test_stop_execution_when_callback_raises_pipeline_stop_requested_with_two_llms(self):
        """
        Run the first LLM and fallback to the second LLM, and then stop execution 
        just before the operation was about to succeed.
        """
        # Arrange
        llm0 = ResponseMockLLM(
            responses=["raise:I'm the first LLM and I'm bad"],
        )
        llm1 = ResponseMockLLM(
            responses=["I'm the last LLM and I'm not supposed to be run"],
        )
        llm_models = LLMModelWithInstance.from_instances([llm0, llm1])

        def should_stop_callback(parameters: ShouldStopCallbackParameters) -> None:
            if parameters.attempt_index == 0:
                # Continue execution by not raising
                pass
            elif parameters.attempt_index == 1:
                # Stop execution by raising PipelineStopRequested
                raise PipelineStopRequested("Stopping execution after second attempt")
            else:
                raise ValueError(f"Unexpected attempt index: {parameters.attempt_index}")
        
        executor = LLMExecutor(llm_models=llm_models, should_stop_callback=should_stop_callback)

        def execute_function(llm: LLM) -> str:
            return llm.complete("Hi").text

        # Act
        with self.assertRaises(PipelineStopRequested) as context:
            executor.run(execute_function)

        # Assert
        self.assertIn("Stopping execution after second attempt", str(context.exception))
        self.assertEqual(executor.attempt_count, 2)
        attempt0 = executor.attempts[0]
        self.assertFalse(attempt0.success)
        self.assertIsNone(attempt0.result)
        self.assertEqual(str(attempt0.exception), "I'm the first LLM and I'm bad")
        attempt1 = executor.attempts[1]
        self.assertTrue(attempt1.success)
        self.assertEqual(attempt1.result, "I'm the last LLM and I'm not supposed to be run")

    def test_exception_inside_should_stop_callback(self):
        """
        The should_stop_callback is supposed to be a function that can raise PipelineStopRequested.
        Exercise what happens when the should_stop_callback raises an exception other than PipelineStopRequested, such as broken database connection.
        """
        # Arrange
        llm0 = ResponseMockLLM(
            responses=["I'm the first LLM and I'm good"],
        )
        llm1 = ResponseMockLLM(
            responses=["I'm the last LLM and I'm never supposed to be run"],
        )
        llm_models = LLMModelWithInstance.from_instances([llm0, llm1])

        def should_stop_callback(parameters: ShouldStopCallbackParameters) -> None:
            raise ValueError("Broken database connection")
        
        executor = LLMExecutor(llm_models=llm_models, should_stop_callback=should_stop_callback)

        def execute_function(llm: LLM) -> str:
            return llm.complete("Hi").text

        # Act
        with self.assertRaises(ValueError) as context:
            executor.run(execute_function)

        # Assert
        self.assertIn("Broken database connection", str(context.exception))
        self.assertEqual(executor.attempt_count, 1)
        attempt0 = executor.attempts[0]
        self.assertTrue(attempt0.success)
        self.assertEqual(attempt0.result, "I'm the first LLM and I'm good")

    def test_raise_pipelinestoprequested_within_execute_function(self):
        """
        Example of what not to do:
        The execute_function is not supposed to raise the PipelineStopRequested exception.
        This test does exactly that, and check that it gets handled properly.
        """
        # Arrange
        llm1 = ResponseMockLLM(
            responses=["I'm 1st LLM"],
        )
        llm2 = ResponseMockLLM(
            responses=["I'm 2nd LLM"],
        )
        llm_models = LLMModelWithInstance.from_instances([llm1, llm2])
        executor = LLMExecutor(llm_models=llm_models)

        def execute_function(llm: LLM) -> str:
            # The PipelineStopRequested is supposed to be raised by the should_stop_callback, not by the execute_function.
            # Here I'm testing that doing the wrong thing gets handled properly.
            # This it stops the execution, and no further execution attempts are made.
            raise PipelineStopRequested("execute function requested pipeline stop")

        # Act
        with self.assertRaises(PipelineStopRequested) as context:
            executor.run(execute_function)

        # Assert
        self.assertIn("execute function requested pipeline stop", str(context.exception))
        self.assertEqual(executor.attempt_count, 0)

    def test_llmexecutor_init_with_no_llms(self):
        """One or more LLMs are supposed to be provided."""
        # Act
        with self.assertRaises(ValueError) as context:
            LLMExecutor(llm_models=[])

        # Assert
        self.assertIn("No LLMs provided", str(context.exception))

    def test_llmexecutor_init_with_junk_callback(self):
        """The callback is supposed to be a function that can raise PipelineStopRequested."""
        # Arrange
        llm_model = LLMModelWithInstance(ResponseMockLLM(responses=["test"]))

        # Act
        with self.assertRaises(TypeError) as context:
            LLMExecutor(llm_models=[llm_model], should_stop_callback="I'm not a function")

        # Assert
        self.assertIn("should_stop_callback must be a function that can raise PipelineStopRequested to stop execution", str(context.exception))

    def test_validate_execute_function1(self):
        """
        Invoke the run() function with a junk execute_function, and check that it detects that it's junk.
        The execute_function is supposed to be a function that takes a LLM parameter.
        """
        # Arrange
        llm_model = LLMModelWithInstance(ResponseMockLLM(responses=["test"]))
        executor = LLMExecutor(llm_models=[llm_model])

        # Act
        with self.assertRaises(TypeError) as context:
            executor.run("I'm not a function")

        # Assert
        self.assertIn("validate_execute_function1: must be a function that takes a LLM parameter", str(context.exception))

    def test_validate_execute_function2(self):
        """
        Invoke the run() function with a junk execute_function, and check that it detects that it's junk.
        The execute_function is supposed to be a function that takes a LLM parameter.
        """
        # Arrange
        llm_model = LLMModelWithInstance(ResponseMockLLM(responses=["test"]))
        executor = LLMExecutor(llm_models=[llm_model])

        def execute_function(a: int, b: int, c: int) -> str:
            raise ValueError("I take the wrong number of parameters, I'm not supposed to be called")

        # Act
        with self.assertRaises(TypeError) as context:
            executor.run(execute_function)

        # Assert
        self.assertIn("validate_execute_function2: must be a function that takes a single parameter", str(context.exception))

    def test_validate_execute_function3(self):
        """
        Invoke the run() function with a junk execute_function, and check that it detects that it's junk.
        The execute_function is supposed to be a function that takes a LLM parameter.
        """
        # Arrange
        llm_model = LLMModelWithInstance(ResponseMockLLM(responses=["test"]))
        executor = LLMExecutor(llm_models=[llm_model])

        def execute_function(wrong_parameter_type: str) -> str:
            raise ValueError("I have the wrong function type signature, I'm not supposed to be called")

        # Act
        with self.assertRaises(TypeError) as context:
            executor.run(execute_function)

        # Assert
        # Update the assertion to match the new, more specific error message.
        expected_error_part_1 = "validate_execute_function3: must be a function that takes a single parameter of type LLM"
        expected_error_part_2 = "but got type"
        
        exception_string = str(context.exception)
        self.assertIn(expected_error_part_1, exception_string)
        self.assertIn(expected_error_part_2, exception_string)
        self.assertIn("<class 'str'>", exception_string) # Be very specific about the type found
        
    def test_validate_execute_function3_with_postponed_annotations(self):
        """
        Exercise what happens when the execute_function has a type hint that is a string,
        `from __future__ import annotations` (PEP 563), which turns type hints into strings at definition time.
        """
        # Arrange
        llm_model = LLMModelWithInstance(ResponseMockLLM(responses=["test"]))
        # Use the NEW, ROBUST LLMExecutor. For this test, we'll assume the
        # main LLMExecutor has been updated. If not, you'd instantiate a
        # patched version here.
        executor = LLMExecutor(llm_models=[llm_model])

        # --- Create a temporary module with `from __future__ import annotations` ---
        # This is the only reliable way to test this feature.
        module_code = """
from __future__ import annotations
from llama_index.core.llms.llm import LLM

def good_function(llm: LLM) -> str:
    return llm.complete("Hi").text

def bad_function(wrong_type: str) -> str:
    return "should not run"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(module_code)
            tmp_path = Path(tmp.name)

        try:
            # Dynamically import the temporary module
            spec = importlib.util.spec_from_file_location("test_module", tmp_path)
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)

            # --- Act & Assert ---

            # 1. Test the GOOD function with the postponed 'LLM' annotation
            # This should PASS validation and run successfully.
            try:
                result = executor.run(test_module.good_function)
                self.assertEqual(result, "test")
            except TypeError as e:
                self.fail(f"Validation incorrectly failed for a valid function with postponed annotations: {e}")

            # 2. Test the BAD function with the postponed 'str' annotation
            # This should FAIL validation.
            with self.assertRaises(TypeError) as context:
                executor.run(test_module.bad_function)

            # Check for a more specific error message from the robust validator
            self.assertIn("validate_execute_function3: must be a function that takes a single parameter of type LLM", str(context.exception))
            self.assertIn("but got type", str(context.exception)) # Example part of the new message

        finally:
            # Clean up the temporary file
            if tmp_path.exists():
                tmp_path.unlink()

    def test_failed_attempt_records_error_id_in_usage_metrics(self):
        """When execute_function raises a raw exception (not LLMChatError),
        the usage metric should still contain an error_id generated by _try_one_attempt."""
        # Arrange
        bad_llm = ResponseMockLLM(responses=["raise:BAD"])
        llm_models = LLMModelWithInstance.from_instances([bad_llm])
        executor = LLMExecutor(llm_models=llm_models)

        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_file = Path(tmp_dir) / "usage_metrics.jsonl"
            set_usage_metrics_path(metrics_file)
            try:
                def execute_function(llm: LLM) -> str:
                    return llm.complete("Hi").text

                # Act
                with self.assertRaises(Exception):
                    executor.run(execute_function)

                # Assert
                lines = metrics_file.read_text().strip().splitlines()
                self.assertEqual(len(lines), 1)
                record = json.loads(lines[0])
                self.assertFalse(record["success"])
                self.assertIn("error_id", record, "error_id must always be present for failed attempts")
                self.assertEqual(len(record["error_id"]), 12, "error_id should be a 12-char hex string")
            finally:
                set_usage_metrics_path(None)

    def test_failed_create_llm_records_error_id_in_usage_metrics(self):
        """When create_llm() fails, the usage metric should still contain an error_id."""
        # Arrange
        class BadLLMModel(LLMModelBase):
            def create_llm(self) -> LLM:
                raise ValueError("Cannot initialize this model")
            def __repr__(self) -> str:
                return "BadLLMModel()"

        executor = LLMExecutor(llm_models=[BadLLMModel()])

        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_file = Path(tmp_dir) / "usage_metrics.jsonl"
            set_usage_metrics_path(metrics_file)
            try:
                def execute_function(llm: LLM) -> str:
                    return llm.complete("Hi").text

                # Act
                with self.assertRaises(Exception):
                    executor.run(execute_function)

                # Assert
                lines = metrics_file.read_text().strip().splitlines()
                self.assertEqual(len(lines), 1)
                record = json.loads(lines[0])
                self.assertFalse(record["success"])
                self.assertIn("error_id", record, "error_id must always be present for failed attempts")
                self.assertEqual(len(record["error_id"]), 12)
            finally:
                set_usage_metrics_path(None)

    def test_llm_chat_error_preserves_existing_error_id(self):
        """When execute_function raises LLMChatError (which already has error_id),
        that error_id should be preserved in the usage metric."""
        from worker_plan_internal.llm_util.llm_errors import LLMChatError

        # Arrange
        good_llm = ResponseMockLLM(responses=["unused"])
        llm_models = LLMModelWithInstance.from_instances([good_llm])
        executor = LLMExecutor(llm_models=llm_models)

        known_error_id = "aabbccdd1122"

        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_file = Path(tmp_dir) / "usage_metrics.jsonl"
            set_usage_metrics_path(metrics_file)
            try:
                def execute_function(llm: LLM) -> str:
                    raise LLMChatError(
                        cause=ValueError("bad json"),
                        error_id=known_error_id,
                    )

                # Act
                with self.assertRaises(Exception):
                    executor.run(execute_function)

                # Assert
                lines = metrics_file.read_text().strip().splitlines()
                self.assertEqual(len(lines), 1)
                record = json.loads(lines[0])
                self.assertEqual(record["error_id"], known_error_id)
            finally:
                set_usage_metrics_path(None)