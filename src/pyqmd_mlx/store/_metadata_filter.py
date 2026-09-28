"""QMD Metadata Filter -- recursive filter AST, strict runtime validation,
and parameterized SQL compilation.

The filter has one canonical, operator-discriminated recursive shape shared
by every search surface (CLI, MCP):

    {"operator": "and", "operands": [...]}
    {"operator": "not", "operand": {...}}
    {"key": "status", "operator": "eq", "value": "published"}

Ported from the Node reference's `src/metadata-filter.ts`. Pure functions --
no `self`, no SQL evaluation in Python. Compilation emits correlated
EXISTS/NOT EXISTS subqueries over `document_metadata_values` with every
user value bound as a parameter -- metadata keys and values are data, never
SQL.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._metadata import METADATA_LIMITS as _EXTRACTION_LIMITS


@dataclass
class MetadataFilterGroup:
    operator: str  # "and" | "or"
    operands: list["MetadataFilter"]


@dataclass
class MetadataFilterNegation:
    operator: str  # "not"
    operand: "MetadataFilter"


@dataclass
class MetadataCondition:
    key: str
    operator: str  # eq/ne/gt/gte/lt/lte/in/nin/all/exists
    value: object


MetadataFilter = MetadataFilterGroup | MetadataFilterNegation | MetadataCondition


class MetadataFilterError(Exception):
    """Raised by parse_metadata_filter with the JSON path of the failing node."""

    def __init__(self, path: str, message: str):
        super().__init__(f"Invalid metadata filter at {path}: {message}")
        self.path = path


# Defensive limits for recursive filters from untrusted callers.
METADATA_FILTER_LIMITS: dict[str, int] = {
    "max_depth": 16,
    "max_nodes": 256,
    "max_group_operands": 32,
    "max_membership_values": 64,
    "max_key_bytes": _EXTRACTION_LIMITS["max_key_bytes"],
    "max_string_length": _EXTRACTION_LIMITS["max_string_length"],
}

_GROUP_OPERATORS = ["and", "or"]
_COMPARISON_OPERATORS = ["eq", "ne", "gt", "gte", "lt", "lte"]
_ORDERED_OPERATORS = ["gt", "gte", "lt", "lte"]
_MEMBERSHIP_OPERATORS = ["in", "nin", "all"]
_CONDITION_OPERATORS = [*_COMPARISON_OPERATORS, *_MEMBERSHIP_OPERATORS, "exists"]
_ALL_OPERATORS = [*_GROUP_OPERATORS, "not", *_CONDITION_OPERATORS]


def parse_metadata_filter(input_value: object) -> MetadataFilter:
    """Strictly validate an untrusted value as a MetadataFilter. Rejects
    unknown operators, unknown properties, operator-incompatible values, and
    inputs exceeding METADATA_FILTER_LIMITS. Canonicalizes membership value
    arrays by de-duplicating while preserving order."""
    state = {"nodes": 0}
    return _parse_filter_node(input_value, "$", 1, state)


def _parse_filter_node(input_value: object, path: str, depth: int, state: dict) -> MetadataFilter:
    if depth > METADATA_FILTER_LIMITS["max_depth"]:
        raise MetadataFilterError(
            path, f"exceeds maximum nesting depth of {METADATA_FILTER_LIMITS['max_depth']}"
        )

    state["nodes"] += 1
    if state["nodes"] > METADATA_FILTER_LIMITS["max_nodes"]:
        raise MetadataFilterError(
            path, f"exceeds maximum of {METADATA_FILTER_LIMITS['max_nodes']} nodes"
        )

    if not isinstance(input_value, dict):
        raise MetadataFilterError(path, "each filter node must be an object")

    operator = input_value.get("operator")
    if not isinstance(operator, str):
        raise MetadataFilterError(path, "missing 'operator' property")

    if operator in _GROUP_OPERATORS:
        return _parse_filter_group(input_value, operator, path, depth, state)
    if operator == "not":
        return _parse_filter_negation(input_value, path, depth, state)
    if operator in _CONDITION_OPERATORS:
        return _parse_filter_condition(input_value, operator, path)

    raise MetadataFilterError(
        path, f"unknown operator '{operator}' -- expected one of: {', '.join(_ALL_OPERATORS)}"
    )


def _parse_filter_group(
    node: dict, operator: str, path: str, depth: int, state: dict
) -> MetadataFilterGroup:
    _reject_unknown_properties(node, ["operator", "operands"], path)

    operands = node.get("operands")
    if not isinstance(operands, list):
        raise MetadataFilterError(path, f"'{operator}' requires an 'operands' array")
    if len(operands) == 0:
        raise MetadataFilterError(path, f"'{operator}' requires a non-empty 'operands' array")
    if len(operands) > METADATA_FILTER_LIMITS["max_group_operands"]:
        raise MetadataFilterError(
            path,
            f"'{operator}' exceeds maximum of {METADATA_FILTER_LIMITS['max_group_operands']} operands",
        )

    return MetadataFilterGroup(
        operator=operator,
        operands=[
            _parse_filter_node(operand, f"{path}.operands[{i}]", depth + 1, state)
            for i, operand in enumerate(operands)
        ],
    )


def _parse_filter_negation(
    node: dict, path: str, depth: int, state: dict
) -> MetadataFilterNegation:
    _reject_unknown_properties(node, ["operator", "operand"], path)
    if "operand" not in node:
        raise MetadataFilterError(path, "'not' requires exactly one 'operand'")
    return MetadataFilterNegation(
        operator="not",
        operand=_parse_filter_node(node["operand"], f"{path}.operand", depth + 1, state),
    )


def _parse_filter_condition(node: dict, operator: str, path: str) -> MetadataCondition:
    _reject_unknown_properties(node, ["key", "operator", "value"], path)

    key = node.get("key")
    if not isinstance(key, str) or len(key) == 0:
        raise MetadataFilterError(path, f"'{operator}' requires a non-empty string 'key'")
    if len(key.encode("utf-8")) > METADATA_FILTER_LIMITS["max_key_bytes"]:
        raise MetadataFilterError(
            path, f"'key' exceeds {METADATA_FILTER_LIMITS['max_key_bytes']} bytes"
        )

    if "value" not in node:
        raise MetadataFilterError(path, f"'{operator}' requires a 'value'")
    value = node["value"]

    if operator == "exists":
        if not isinstance(value, bool):
            raise MetadataFilterError(f"{path}.value", "'exists' requires a boolean value")
        return MetadataCondition(key=key, operator=operator, value=value)

    if operator in _MEMBERSHIP_OPERATORS:
        return MetadataCondition(
            key=key, operator=operator, value=_parse_membership_values(value, operator, path)
        )

    scalar = _parse_scalar_value(value, f"{path}.value")
    if operator in _ORDERED_OPERATORS and isinstance(scalar, bool):
        raise MetadataFilterError(
            f"{path}.value", f"'{operator}' requires a string or number value"
        )
    return MetadataCondition(key=key, operator=operator, value=scalar)


def _parse_membership_values(value: object, operator: str, path: str) -> list:
    if not isinstance(value, list):
        raise MetadataFilterError(f"{path}.value", f"'{operator}' requires an array value")
    if len(value) == 0:
        raise MetadataFilterError(f"{path}.value", f"'{operator}' requires a non-empty array value")
    if len(value) > METADATA_FILTER_LIMITS["max_membership_values"]:
        raise MetadataFilterError(
            f"{path}.value",
            f"'{operator}' exceeds maximum of {METADATA_FILTER_LIMITS['max_membership_values']} values",
        )

    scalars = [_parse_scalar_value(v, f"{path}.value[{i}]") for i, v in enumerate(value)]

    if all(isinstance(s, str) for s in scalars):
        return list(dict.fromkeys(scalars))
    if all(isinstance(s, bool) for s in scalars):
        return list(dict.fromkeys(scalars))
    if all(isinstance(s, (int, float)) and not isinstance(s, bool) for s in scalars):
        return list(dict.fromkeys(scalars))
    raise MetadataFilterError(
        f"{path}.value", f"'{operator}' requires a homogeneous array of one scalar type"
    )


def _parse_scalar_value(value: object, path: str):
    # bool must be checked before int/float -- bool is an int subclass.
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if len(value) > METADATA_FILTER_LIMITS["max_string_length"]:
            raise MetadataFilterError(
                path, f"string exceeds {METADATA_FILTER_LIMITS['max_string_length']} characters"
            )
        return value
    if isinstance(value, (int, float)):
        import math

        if isinstance(value, float) and not math.isfinite(value):
            raise MetadataFilterError(path, "numbers must be finite")
        return value
    raise MetadataFilterError(path, "expected a string, number, or boolean")


def _reject_unknown_properties(node: dict, allowed: list[str], path: str) -> None:
    for prop in node:
        if prop not in allowed:
            raise MetadataFilterError(
                path, f"unknown property '{prop}' -- allowed: {', '.join(allowed)}"
            )


@dataclass
class CompiledMetadataFilter:
    sql: str
    params: list


def compile_metadata_filter(
    filter_: MetadataFilter, documents_alias: str
) -> CompiledMetadataFilter:
    """Compile a validated filter into one parameterized SQL predicate
    correlated against a documents-table alias (e.g. `d`). All keys and
    values are bound parameters. The caller is responsible for restricting
    the surrounding query to active documents with current, error-free
    metadata extraction."""
    params: list = []
    sql = _compile_filter_node(filter_, documents_alias, params)
    return CompiledMetadataFilter(sql=sql, params=params)


def _compile_filter_node(filter_: MetadataFilter, alias: str, params: list) -> str:
    if isinstance(filter_, MetadataFilterGroup):
        joiner = " AND " if filter_.operator == "and" else " OR "
        return (
            "("
            + joiner.join(_compile_filter_node(o, alias, params) for o in filter_.operands)
            + ")"
        )

    if isinstance(filter_, MetadataFilterNegation):
        return f"NOT {_compile_filter_node(filter_.operand, alias, params)}"

    op = filter_.operator

    if op == "exists":
        params.append(filter_.key)
        exists_sql = _build_value_exists_sql(alias, "mv.key = ?")
        return exists_sql if filter_.value else f"NOT {exists_sql}"

    if op in ("eq", "gt", "gte", "lt", "lte"):
        sql_operator = {"eq": "=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}[op]
        value_type = _value_type_of(filter_.value)
        column = _value_column_of(filter_.value)
        params.append(filter_.key)
        params.append(_bind_scalar(filter_.value))
        return _build_value_exists_sql(
            alias, f"mv.key = ? AND mv.value_type = '{value_type}' AND mv.{column} {sql_operator} ?"
        )

    if op == "ne":
        # Key must have at least one same-type value, and no same-type value
        # may equal the operand. Missing keys and type mismatches do not match.
        value_type = _value_type_of(filter_.value)
        column = _value_column_of(filter_.value)
        params.append(filter_.key)
        present_sql = _build_value_exists_sql(
            alias, f"mv.key = ? AND mv.value_type = '{value_type}'"
        )
        params.append(filter_.key)
        params.append(_bind_scalar(filter_.value))
        equal_sql = _build_value_exists_sql(
            alias, f"mv.key = ? AND mv.value_type = '{value_type}' AND mv.{column} = ?"
        )
        return f"({present_sql} AND NOT {equal_sql})"

    if op in ("in", "nin"):
        value_type = _value_type_of(filter_.value[0])
        column = _value_column_of(filter_.value[0])
        placeholders = ", ".join("?" for _ in filter_.value)
        if op == "in":
            params.append(filter_.key)
            params.extend(_bind_scalar(v) for v in filter_.value)
            return _build_value_exists_sql(
                alias,
                f"mv.key = ? AND mv.value_type = '{value_type}' AND mv.{column} IN ({placeholders})",
            )
        params.append(filter_.key)
        present_sql = _build_value_exists_sql(
            alias, f"mv.key = ? AND mv.value_type = '{value_type}'"
        )
        params.append(filter_.key)
        params.extend(_bind_scalar(v) for v in filter_.value)
        member_sql = _build_value_exists_sql(
            alias,
            f"mv.key = ? AND mv.value_type = '{value_type}' AND mv.{column} IN ({placeholders})",
        )
        return f"({present_sql} AND NOT {member_sql})"

    if op == "all":
        value_type = _value_type_of(filter_.value[0])
        column = _value_column_of(filter_.value[0])
        member_sqls = []
        for element in filter_.value:
            params.append(filter_.key)
            params.append(_bind_scalar(element))
            member_sqls.append(
                _build_value_exists_sql(
                    alias, f"mv.key = ? AND mv.value_type = '{value_type}' AND mv.{column} = ?"
                )
            )
        return "(" + " AND ".join(member_sqls) + ")"

    raise AssertionError(f"unhandled filter operator: {op}")


def _build_value_exists_sql(alias: str, condition_sql: str) -> str:
    return f"EXISTS (SELECT 1 FROM document_metadata_values mv WHERE mv.document_id = {alias}.id AND {condition_sql})"


def _value_type_of(scalar) -> str:
    # bool must be checked before int/float -- bool is an int subclass.
    if isinstance(scalar, bool):
        return "boolean"
    if isinstance(scalar, str):
        return "string"
    return "number"


def _value_column_of(scalar) -> str:
    if isinstance(scalar, str):
        return "text_value"
    if isinstance(scalar, bool):
        return "boolean_value"
    return "number_value"


def _bind_scalar(scalar):
    if isinstance(scalar, bool):
        return 1 if scalar else 0
    return scalar
