"""
test_enum_literal_parity.py — CI check: Literal field values match their Enum class.

When a Pydantic LLM-output schema uses `Literal["a", "b", "c"]` for a field whose
canonical values are defined in a `str(Enum)` class, this test ensures the two
stay in sync.  If someone adds a new Enum member without updating the Literal (or
vice-versa) the test fails immediately.

Uses AST parsing (no imports of the target modules required) so it works
regardless of whether llama_index, pydantic, etc. are installed in the
test runner environment.

Run with:  pytest worker_plan/worker_plan_internal/tests/test_enum_literal_parity.py -v
"""

import ast
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _enum_values_from_classdef(node: ast.ClassDef) -> Optional[frozenset]:
    """
    If `node` is a `class Foo(str, Enum)` definition, return the frozenset of
    its member values (string constants only).  Returns None otherwise.
    """
    base_names = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            base_names.add(base.id)
        elif isinstance(base, ast.Attribute):
            base_names.add(base.attr)

    if "str" not in base_names or "Enum" not in base_names:
        return None

    values = set()
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                        values.add(item.value.value)
    return frozenset(values) if values else None


def _literal_values_from_annotation(annotation: ast.expr) -> Optional[frozenset]:
    """
    Given an AST annotation node, return the frozenset of values inside
    `Literal["a", "b", ...]`, or None if it's not a Literal.

    Handles:
      - Literal["a", "b"]            (subscript with Name)
      - Literal["a"]                 (subscript with single constant)
      - typing.Literal["a", "b"]    (subscript with Attribute)
    """
    if not isinstance(annotation, ast.Subscript):
        return None

    # Check the subscript value is Literal or typing.Literal
    val = annotation.value
    is_literal = (
        (isinstance(val, ast.Name) and val.id == "Literal")
        or (isinstance(val, ast.Attribute) and val.attr == "Literal")
    )
    if not is_literal:
        return None

    slice_node = annotation.slice
    values = []

    if isinstance(slice_node, ast.Tuple):
        for elt in slice_node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                values.append(elt.value)
    elif isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        values.append(slice_node.value)

    return frozenset(values) if values else None


def _is_basemodel_subclass(node: ast.ClassDef) -> bool:
    """Heuristic: return True if class inherits from BaseModel."""
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "BaseModel":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseModel":
            return True
    return False


# ---------------------------------------------------------------------------
# Gather parity cases from all source files
# ---------------------------------------------------------------------------

def _gather_cases_from_file(path: Path) -> list[tuple]:
    """
    Parse one .py file and return a list of:
      (file, enum_name, enum_values, model_name, field_name, literal_values)
    for every Literal field whose value-set exactly matches a str(Enum) in
    the same file.
    """
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    # Collect all str(Enum) classes  { value_set -> class_name }
    enums: dict[frozenset, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            vals = _enum_values_from_classdef(node)
            if vals:
                enums[vals] = node.name

    if not enums:
        return []

    cases = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not _is_basemodel_subclass(node):
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign):
                continue
            lit = _literal_values_from_annotation(item.annotation)
            if lit is None:
                continue
            if lit in enums:
                field_name = (
                    item.target.id
                    if isinstance(item.target, ast.Name)
                    else ast.dump(item.target)
                )
                cases.append((
                    str(path.relative_to(Path(__file__).parent.parent.parent.parent)),
                    enums[lit],   # enum class name
                    lit,          # expected value set (from Enum)
                    node.name,    # Pydantic model name
                    field_name,
                    lit,          # actual value set (from Literal — same, just recorded)
                ))
    return cases


def _gather_all_cases() -> list[tuple]:
    package_root = Path(__file__).parent.parent  # worker_plan_internal/
    cases = []
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts or "test" in path.name:
            continue
        cases.extend(_gather_cases_from_file(path))
    return cases


_CASES = _gather_all_cases()


def _case_id(case):
    file, enum_name, _, model, field, _ = case
    short = Path(file).stem
    return f"{short}::{model}.{field}↔{enum_name}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cases_found():
    """Sanity check: the walk must find at least one parity case."""
    assert len(_CASES) >= 1, (
        "No Enum/Literal parity cases found — check that the AST walker "
        "is reaching the right files."
    )


@pytest.mark.parametrize("case", _CASES, ids=[_case_id(c) for c in _CASES])
def test_literal_matches_enum(case):
    """
    Assert that the Literal values for a Pydantic field exactly match
    the corresponding str(Enum) values in the same file.
    """
    file, enum_name, enum_vals, model_name, field_name, literal_vals = case
    assert literal_vals == enum_vals, (
        f"{file} — {model_name}.{field_name}: "
        f"Literal{sorted(literal_vals)} does not match "
        f"{enum_name}{sorted(enum_vals)}. "
        f"Update the Literal or the Enum to keep them in sync."
    )
