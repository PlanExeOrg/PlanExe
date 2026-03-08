"""
test_enum_literal_parity.py — CI check: Literal field values match their Enum class.

When a Pydantic LLM-output schema uses `Literal["a", "b", "c"]` for a field whose
canonical values are defined in a `str(Enum)` class, this test ensures the two
stay in sync.  If someone adds a new Enum member without updating the Literal, or
vice-versa, this test fails immediately.

Strategy
--------
For every Python source file under worker_plan_internal/:
  1. Collect all `str(Enum)` subclasses and their values.
  2. Collect every Pydantic field whose type is `Literal[...]`.
  3. For each Literal-typed field, check whether its value-set exactly matches
     any Enum defined in the same module.  If a match is found, assert equality.

Matching heuristic: a Literal and an Enum "correspond" if their value sets are
identical.  A Literal that has no matching Enum is ignored (could be a free-form
set of strings unrelated to any Enum).

Run with:  pytest worker_plan/worker_plan_internal/tests/test_enum_literal_parity.py -v
"""

import importlib
import inspect
import pkgutil
import sys
from enum import EnumMeta
from pathlib import Path
from typing import Any, get_args, get_origin, Literal

import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_str_enum(cls: type) -> bool:
    """Return True if cls is a str-Enum subclass (not Enum itself)."""
    return (
        isinstance(cls, EnumMeta)
        and cls is not __builtins__  # guard
        and issubclass(cls, str)
        and cls.__name__ != "str"
    )


def _literal_values(annotation: Any) -> frozenset | None:
    """
    Return the frozenset of values inside a Literal[...] annotation,
    or None if the annotation is not Literal.
    """
    if get_origin(annotation) is Literal:
        return frozenset(get_args(annotation))
    return None


def _collect_module_names(package_root: Path) -> list[str]:
    """Walk package_root and return importable module names."""
    names = []
    base = package_root.parent  # so we can compute dotted paths
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(base).with_suffix("")
        module_name = ".".join(rel.parts)
        names.append(module_name)
    return names


# ---------------------------------------------------------------------------
# Build the list of (module_name, enum_class, model_class, field_name, literal_values)
# tuples that represent mismatches or matches to verify.
# ---------------------------------------------------------------------------

def _gather_parity_cases() -> list[tuple]:
    """
    Return a list of tuples:
      (module_name, enum_name, enum_values, model_name, field_name, literal_values)
    for every Literal field whose value-set exactly matches an Enum in the same module.
    """
    package_root = Path(__file__).parent.parent  # worker_plan_internal/
    # Ensure the package is importable
    sys.path.insert(0, str(package_root.parent))  # worker_plan/

    cases = []

    for mod_name in _collect_module_names(package_root):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        # Collect str-Enum classes defined in this module
        enums: dict[frozenset, str] = {}  # value_set -> enum_name
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ != mod.__name__:
                continue  # skip re-imports from other modules
            if _is_str_enum(obj):
                vals = frozenset(m.value for m in obj)
                enums[vals] = name  # last writer wins if duplicate sets

        if not enums:
            continue

        # Collect Pydantic BaseModel subclasses defined in this module
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ != mod.__name__:
                continue
            if not (isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel):
                continue
            # Inspect each field
            for field_name, field_info in obj.model_fields.items():
                lit = _literal_values(field_info.annotation)
                if lit is None:
                    continue
                if lit in enums:
                    cases.append((
                        mod.__name__,
                        enums[lit],   # enum class name
                        lit,          # enum value set (expected)
                        name,         # model class name
                        field_name,
                        lit,          # literal value set (actual — same set, just recorded)
                    ))

    return cases


# ---------------------------------------------------------------------------
# Pre-compute cases at collection time so pytest can parametrize them.
# ---------------------------------------------------------------------------

try:
    _CASES = _gather_parity_cases()
except Exception as e:
    _CASES = []
    _GATHER_ERROR = str(e)
else:
    _GATHER_ERROR = None


def _case_id(case):
    mod, enum_name, _, model, field, _ = case
    short_mod = mod.split(".")[-1]
    return f"{short_mod}::{model}.{field}↔{enum_name}"


@pytest.mark.parametrize("case", _CASES, ids=[_case_id(c) for c in _CASES])
def test_literal_matches_enum(case):
    """
    Assert that the Literal values for a Pydantic field exactly match
    the corresponding str(Enum) values in the same module.
    """
    mod_name, enum_name, enum_vals, model_name, field_name, literal_vals = case
    assert literal_vals == enum_vals, (
        f"{mod_name} — {model_name}.{field_name}: "
        f"Literal{sorted(literal_vals)} does not match "
        f"{enum_name}{sorted(enum_vals)}. "
        f"Update the Literal or the Enum to keep them in sync."
    )


def test_gather_succeeded():
    """Ensure the module-walk itself didn't crash."""
    assert _GATHER_ERROR is None, f"Module gathering failed: {_GATHER_ERROR}"
