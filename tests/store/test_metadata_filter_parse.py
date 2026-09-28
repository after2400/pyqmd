import pytest

from pyqmd_mlx.store._metadata_filter import (
    MetadataCondition,
    MetadataFilterError,
    MetadataFilterGroup,
    MetadataFilterNegation,
    parse_metadata_filter,
)


def test_parses_simple_eq_condition():
    result = parse_metadata_filter({"key": "status", "operator": "eq", "value": "published"})
    assert result == MetadataCondition(key="status", operator="eq", value="published")


@pytest.mark.parametrize("operator", ["eq", "ne", "gt", "gte", "lt", "lte"])
def test_parses_comparison_operators_with_string_value(operator):
    result = parse_metadata_filter({"key": "k", "operator": operator, "value": "v"})
    assert result == MetadataCondition(key="k", operator=operator, value="v")


@pytest.mark.parametrize("operator", ["gt", "gte", "lt", "lte"])
def test_ordered_operators_reject_boolean_value(operator):
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter({"key": "k", "operator": operator, "value": True})


@pytest.mark.parametrize("operator", ["in", "nin", "all"])
def test_parses_membership_operators(operator):
    result = parse_metadata_filter({"key": "topics", "operator": operator, "value": ["a", "b"]})
    assert result == MetadataCondition(key="topics", operator=operator, value=["a", "b"])


def test_membership_dedups_while_preserving_order():
    result = parse_metadata_filter({"key": "topics", "operator": "in", "value": ["b", "a", "b"]})
    assert result.value == ["b", "a"]


def test_membership_requires_non_empty_array():
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter({"key": "topics", "operator": "in", "value": []})


def test_membership_requires_homogeneous_array():
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter({"key": "topics", "operator": "in", "value": ["a", 1]})


def test_membership_rejects_oversized_array():
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter({"key": "topics", "operator": "in", "value": list(range(65))})


def test_parses_exists_true():
    result = parse_metadata_filter({"key": "status", "operator": "exists", "value": True})
    assert result == MetadataCondition(key="status", operator="exists", value=True)


def test_exists_requires_boolean_value():
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter({"key": "status", "operator": "exists", "value": "yes"})


def test_parses_and_group():
    result = parse_metadata_filter(
        {
            "operator": "and",
            "operands": [
                {"key": "status", "operator": "eq", "value": "published"},
                {"key": "priority", "operator": "gt", "value": 2},
            ],
        }
    )
    assert isinstance(result, MetadataFilterGroup)
    assert result.operator == "and"
    assert len(result.operands) == 2


def test_group_requires_non_empty_operands():
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter({"operator": "or", "operands": []})


def test_group_rejects_oversized_operands():
    operand = {"key": "k", "operator": "eq", "value": "v"}
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter({"operator": "or", "operands": [operand] * 33})


def test_parses_not_negation():
    result = parse_metadata_filter(
        {"operator": "not", "operand": {"key": "status", "operator": "eq", "value": "draft"}}
    )
    assert isinstance(result, MetadataFilterNegation)
    assert result.operator == "not"


def test_not_requires_exactly_one_operand():
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter({"operator": "not"})


def test_rejects_unknown_operator():
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter({"key": "k", "operator": "bogus", "value": "v"})


def test_rejects_unknown_property():
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter({"key": "k", "operator": "eq", "value": "v", "extra": 1})


def test_rejects_non_object_node():
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter("not-an-object")


def test_rejects_missing_operator():
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter({"key": "k", "value": "v"})


def test_rejects_excessive_nesting_depth():
    node = {"key": "k", "operator": "eq", "value": "v"}
    for _ in range(17):
        node = {"operator": "not", "operand": node}
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter(node)


def test_rejects_excessive_node_count():
    operand = {"key": "k", "operator": "eq", "value": "v"}
    node = {"operator": "or", "operands": [operand] * 32}
    wrapped = {"operator": "and", "operands": [node] * 8}
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter(wrapped)


def test_rejects_oversized_key():
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter({"key": "k" * 129, "operator": "eq", "value": "v"})


def test_rejects_oversized_string_value():
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter({"key": "k", "operator": "eq", "value": "x" * 1025})


def test_rejects_non_finite_number_value():
    with pytest.raises(MetadataFilterError):
        parse_metadata_filter({"key": "k", "operator": "eq", "value": float("inf")})


def test_metadata_filter_error_message_includes_path():
    with pytest.raises(MetadataFilterError) as exc_info:
        parse_metadata_filter(
            {"operator": "and", "operands": [{"key": "k", "operator": "bogus", "value": 1}]}
        )
    assert "$.operands[0]" in str(exc_info.value)
