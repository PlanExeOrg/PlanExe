"""
flat_schema_model.py — Pydantic BaseModel subclass that produces flat JSON schemas.

LM Studio's MLX backend (Outlines grammar compiler) cannot resolve $ref/$defs
references in JSON schemas and returns empty content when they are present.
Python Enum fields in Pydantic models generate $defs + $ref patterns.

FlatSchemaModel overrides model_json_schema() to inline all $defs references
before the schema reaches the LLM adapter. Enum types and all other type
annotations remain unchanged — only the serialized schema is affected.

Usage:
    from llm_util.flat_schema_model import FlatSchemaModel

    class MyOutput(FlatSchemaModel):
        purpose: PlanPurpose = Field(...)  # Enum stays; schema is flat
"""

import copy
from typing import Any
from pydantic import BaseModel


def _inline_refs(schema: dict) -> dict:
    """
    Resolve all $ref references in a JSON schema by inlining $defs.

    Recursively replaces {"$ref": "#/$defs/Foo"} with the definition of Foo,
    then removes the top-level $defs entry.
    """
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"]  # e.g. "#/$defs/PlanPurpose"
                ref_name = ref_path.split("/")[-1]
                if ref_name in defs:
                    return resolve(copy.deepcopy(defs[ref_name]))
                # Unknown $ref — leave as-is to avoid data loss
                return obj
            return {k: resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve(item) for item in obj]
        return obj

    return resolve(schema)


class FlatSchemaModel(BaseModel):
    """
    Pydantic BaseModel that emits flat JSON schemas (no $defs/$ref).

    Inherit from this instead of BaseModel for any Pydantic class used as a
    structured LLM output schema where the target LLM backend cannot resolve
    JSON Schema $ref references (e.g. LM Studio MLX / Outlines).
    """

    @classmethod
    def model_json_schema(cls, **kwargs):
        raw = super().model_json_schema(**kwargs)
        return _inline_refs(raw)
