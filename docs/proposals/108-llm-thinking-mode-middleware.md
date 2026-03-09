# Proposal: Thinking Mode Control via Request-Level Middleware

**Alternative perspective:** Decouple thinking mode from model identity—treat it as a **pipeline concern** that transforms request/execution at the LLMExecutor level.

---

## Problem

Currently:
- Thinking suppression is scattered across task code (model-specific prompt manipulation, `/no_think` strings, etc.)
- No centralized way to request thinking modes (`none|low|default`) at the task level
- Each task must know which models support which thinking modes

**Goal:** Tasks request a thinking mode without knowing the model; the executor handles translation.

---

## Proposed Solution: Middleware Transformation Layer

Instead of adding a `thinking_mode` field to each model in `llm_config`, introduce a **thinking capability registry** and **transformation middleware** that sits in the `LLMExecutor`.

### 1. **Thinking Capability Registry** (New)

Create a lightweight registry mapping models to their thinking capabilities:

```python
# worker_plan_internal/llm_util/thinking_modes.py

from enum import Enum
from dataclasses import dataclass
from typing import Callable, Optional

class ThinkingMode(Enum):
    DEFAULT = "default"   # Model's native thinking behavior
    LOW = "low"           # Reduced/disabled thinking (suppress where possible)
    NONE = "none"         # Completely suppress thinking

@dataclass
class ThinkingCapability:
    """Describes how a model handles thinking modes."""
    model_id: str
    supports_thinking: bool  # Can the model do extended thinking?
    supports_suppression: bool  # Can we reliably suppress thinking?
    suppression_method: Optional[str]  # Method: "parameter", "prompt_instruction", "both"
    
class ThinkingRegistry:
    """
    Centralized registry of thinking capabilities per model.
    Allows querying which models support a given thinking mode.
    """
    
    _REGISTRY = {
        "openai-o1": ThinkingCapability(
            model_id="openai-o1",
            supports_thinking=True,
            supports_suppression=False,  # o1 always thinks
            suppression_method=None,
        ),
        "openai-gpt-4": ThinkingCapability(
            model_id="openai-gpt-4",
            supports_thinking=False,
            supports_suppression=False,
            suppression_method=None,
        ),
        "qwen-reasoning": ThinkingCapability(
            model_id="qwen-reasoning",
            supports_thinking=True,
            supports_suppression=True,
            suppression_method="prompt_instruction",  # Use `/no_think` instruction
        ),
        # ... etc
    }
    
    @classmethod
    def get_capability(cls, model_id: str) -> ThinkingCapability:
        """Get capability info for a model."""
        return cls._REGISTRY.get(model_id)
    
    @classmethod
    def supports_mode(cls, model_id: str, mode: ThinkingMode) -> bool:
        """Check if a model can be used with a given thinking mode."""
        cap = cls.get_capability(model_id)
        if cap is None:
            return True  # Unknown model—assume compatible
        
        if mode == ThinkingMode.DEFAULT:
            return True  # All models support default behavior
        elif mode == ThinkingMode.NONE:
            return cap.supports_suppression
        elif mode == ThinkingMode.LOW:
            return cap.supports_suppression or cap.supports_thinking
        return True
```

### 2. **Thinking Transformation Middleware** (In LLMExecutor)

Add a transformation layer that converts a task-level thinking request into model-specific execution:

```python
# In worker_plan_internal/llm_util/llm_executor.py

@dataclass
class LLMExecutionRequest:
    """Request to execute LLM with thinking mode specified."""
    thinking_mode: ThinkingMode = ThinkingMode.DEFAULT
    # ... other params

class ThinkingTransformer:
    """
    Applies thinking mode transformations to execution.
    Handles model-specific suppression logic.
    """
    
    def __init__(self, thinking_mode: ThinkingMode):
        self.thinking_mode = thinking_mode
    
    def should_apply_to_model(self, model_id: str) -> bool:
        """Check if this transformer applies to the given model."""
        return ThinkingRegistry.supports_mode(model_id, self.thinking_mode)
    
    def transform_llm_kwargs(self, model_id: str, kwargs: dict) -> dict:
        """
        Modify LLM arguments based on thinking mode.
        E.g., add suppress_thinking param, adjust temperature, etc.
        """
        if self.thinking_mode == ThinkingMode.NONE:
            # For models with parameter-based suppression
            cap = ThinkingRegistry.get_capability(model_id)
            if cap and cap.suppression_method in ("parameter", "both"):
                kwargs["suppress_thinking"] = True
        
        return kwargs
    
    def transform_prompt(self, model_id: str, prompt: str) -> str:
        """
        Modify prompt based on thinking mode & model.
        E.g., inject `/no_think` for Qwen.
        """
        if self.thinking_mode == ThinkingMode.NONE:
            cap = ThinkingRegistry.get_capability(model_id)
            if cap and cap.suppression_method in ("prompt_instruction", "both"):
                if "qwen" in model_id.lower():
                    prompt = f"{prompt.strip()}\n/no_think"
        
        return prompt

class LLMExecutor:
    """Enhanced executor with thinking mode support."""
    
    def __init__(
        self,
        llm_models: list[LLMModelBase],
        thinking_mode: ThinkingMode = ThinkingMode.DEFAULT,
        should_stop_callback: Optional[Callable] = None,
    ):
        self.llm_models = llm_models
        self.thinking_mode = thinking_mode
        self.should_stop_callback = should_stop_callback
        self.transformer = ThinkingTransformer(thinking_mode)
        self.attempts: List[LLMAttempt] = []
    
    def run(self, execute_function: Callable[[LLM], Any]):
        """
        Run execute_function, attempting models in priority order.
        Apply thinking mode transformations.
        """
        self.attempts = []
        overall_start_time = time.perf_counter()
        
        for index, llm_model in enumerate(self.llm_models):
            model_id = getattr(llm_model, "name", llm_model.__class__.__name__)
            
            # Skip models that don't support the requested thinking mode
            if not self.transformer.should_apply_to_model(model_id):
                logger.debug(f"Skipping {model_id}—doesn't support {self.thinking_mode.value}")
                continue
            
            attempt = self._try_one_attempt_with_thinking(llm_model, execute_function)
            self.attempts.append(attempt)
            
            self._check_stop_callback(attempt, overall_start_time, index)
            if attempt.success:
                return attempt.result
        
        self._raise_final_exception()
    
    def _try_one_attempt_with_thinking(
        self,
        llm_model: LLMModelBase,
        execute_function: Callable[[LLM], Any],
    ) -> LLMAttempt:
        """
        Execute with thinking mode transformations applied.
        """
        attempt_start_time = time.perf_counter()
        try:
            llm = llm_model.create_llm()
            model_id = getattr(llm_model, "name", "unknown")
            
            # Apply thinking mode transformations
            if hasattr(llm, 'kwargs'):
                llm.kwargs = self.transformer.transform_llm_kwargs(model_id, llm.kwargs)
            
            # Execute wrapped in prompt transformation
            def wrapped_execute(llm: LLM) -> Any:
                # Can't easily transform the prompt here without breaking the interface
                # So we rely on parameter-based suppression; prompt-based handled elsewhere
                return execute_function(llm)
            
            result = wrapped_execute(llm)
            duration = time.perf_counter() - attempt_start_time
            
            logger.info(
                f"LLMExecutor succeeded with {model_id} "
                f"(thinking_mode={self.thinking_mode.value}). "
                f"Duration: {duration:.2f}s"
            )
            return LLMAttempt(
                stage='execute',
                llm_model=llm_model,
                success=True,
                duration=duration,
                result=result,
            )
        except Exception as e:
            duration = time.perf_counter() - attempt_start_time
            logger.error(
                f"LLMExecutor failed with {getattr(llm_model, 'name', 'unknown')}: {e}"
            )
            return LLMAttempt(
                stage='execute',
                llm_model=llm_model,
                success=False,
                duration=duration,
                exception=e,
            )
```

### 3. **Task Integration** (Usage)

Tasks now request thinking modes without model awareness:

```python
# In ReviewPlan or any task

def execute(
    llm_executor: LLMExecutor,
    thinking_mode: ThinkingMode = ThinkingMode.DEFAULT,  # NEW PARAM
    document: str,
) -> ReviewPlan:
    """Execute plan review with requested thinking mode."""
    
    # Create executor with thinking mode
    executor_with_thinking = LLMExecutor(
        llm_models=llm_executor.llm_models,
        thinking_mode=thinking_mode,  # Pass through
    )
    
    def execute_review(llm: LLM) -> ChatResponse:
        # Your existing review logic
        return llm.chat([...])
    
    return executor_with_thinking.run(execute_review)
```

### 4. **Pipeline Thread-Through**

Pass `thinking_mode` through the entire request pipeline:

```
Frontend Request → Worker Task → LLMExecutor
  ├─ thinking_mode: ThinkingMode.LOW
  ├─ Executor filters models by capability
  ├─ Transformer applies model-specific suppression
  └─ Result
```

---

## Benefits

1. **Separation of Concerns**: Thinking mode logic isolated in middleware, not scattered across tasks
2. **Extensible**: New models/thinking methods added to registry without touching task code
3. **Model Agnostic**: Tasks don't need to know model names or capabilities
4. **Backward Compatible**: DEFAULT mode works like today; other modes opt-in
5. **Testable**: Registry and transformer easily unit-tested independently
6. **Observable**: Logging shows which models were skipped/used for which mode

---

## Migration Path

1. **Phase 1**: Implement `ThinkingRegistry` + `ThinkingTransformer` alongside existing `LLMExecutor`
2. **Phase 2**: Add `thinking_mode` parameter to common tasks (ReviewPlan, etc.)
3. **Phase 3**: Thread `thinking_mode` through frontend/API layer
4. **Phase 4**: Populate registry as model capabilities become clear

---

## Open Questions

- How to handle prompt-based transformations cleanly without breaking the `execute_function` interface?
  - **Answer**: Use a wrapper in the task or push into `LLMFactory` at instantiation time
- Should thinking mode be per-task-invocation or per-model in llm_config?
  - **Answer**: Per-invocation for flexibility; llm_config defines capabilities, not defaults
- Cost impact of thinking modes—should we track separately?
  - **Answer**: Yes; `model_token_metrics` already has `thinking_tokens` field

---

## Appendix: Why Not Egon's Per-Model Approach?

This design differs by:
- **Not embedding thinking mode in llm_config**: Keeps model config focused on provider/auth/pricing
- **Centralizing logic in executor**: Single source of truth for how modes are applied
- **Filtering by capability**: Automatically skips incompatible models without hardcoding model names in tasks
- **Middleware pattern**: Cleaner than task-by-task prompt manipulation

Both work; this trades simplicity for cleaner separation and better extensibility.
