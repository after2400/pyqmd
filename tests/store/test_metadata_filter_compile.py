import sqlite3

from pyqmd_mlx.store._metadata_filter import compile_metadata_filter, parse_metadata_filter


def _make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE documents (id INTEGER PRIMARY KEY);
        CREATE TABLE document_metadata_values (
            document_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            value_type TEXT NOT NULL,
            text_value TEXT,
            number_value REAL,
            boolean_value INTEGER
        );
        """
    )
    return conn


def _insert(conn, document_id, key, value, ordinal=0):
    if isinstance(value, bool):
        conn.execute("INSERT INTO documents (id) VALUES (?) ON CONFLICT DO NOTHING", (document_id,))
        conn.execute(
            "INSERT INTO document_metadata_values (document_id, key, ordinal, value_type, boolean_value) "
            "VALUES (?, ?, ?, 'boolean', ?)",
            (document_id, key, ordinal, 1 if value else 0),
        )
    elif isinstance(value, str):
        conn.execute("INSERT INTO documents (id) VALUES (?) ON CONFLICT DO NOTHING", (document_id,))
        conn.execute(
            "INSERT INTO document_metadata_values (document_id, key, ordinal, value_type, text_value) "
            "VALUES (?, ?, ?, 'string', ?)",
            (document_id, key, ordinal, value),
        )
    else:
        conn.execute("INSERT INTO documents (id) VALUES (?) ON CONFLICT DO NOTHING", (document_id,))
        conn.execute(
            "INSERT INTO document_metadata_values (document_id, key, ordinal, value_type, number_value) "
            "VALUES (?, ?, ?, 'number', ?)",
            (document_id, key, ordinal, value),
        )
    conn.commit()


def _matching_ids(conn, filter_dict):
    compiled = compile_metadata_filter(parse_metadata_filter(filter_dict), "d")
    rows = conn.execute(
        f"SELECT d.id FROM documents d WHERE {compiled.sql}", compiled.params
    ).fetchall()
    return sorted(r["id"] for r in rows)


def test_eq_matches_exact_string_value():
    conn = _make_conn()
    _insert(conn, 1, "status", "published")
    _insert(conn, 2, "status", "draft")
    assert _matching_ids(conn, {"key": "status", "operator": "eq", "value": "published"}) == [1]


def test_eq_does_not_match_across_value_types():
    conn = _make_conn()
    _insert(conn, 1, "count", "3")  # stored as string
    assert _matching_ids(conn, {"key": "count", "operator": "eq", "value": 3}) == []


def test_ne_requires_present_same_type_value_that_differs():
    conn = _make_conn()
    _insert(conn, 1, "status", "published")
    _insert(conn, 2, "status", "draft")
    # doc 3 has no `status` key at all -- ne must NOT match it.
    conn.execute("INSERT INTO documents (id) VALUES (3)")
    conn.commit()
    assert _matching_ids(conn, {"key": "status", "operator": "ne", "value": "published"}) == [2]


def test_gt_compares_numbers():
    conn = _make_conn()
    _insert(conn, 1, "priority", 1)
    _insert(conn, 2, "priority", 5)
    assert _matching_ids(conn, {"key": "priority", "operator": "gt", "value": 2}) == [2]


def test_in_matches_membership():
    conn = _make_conn()
    _insert(conn, 1, "status", "published")
    _insert(conn, 2, "status", "archived")
    _insert(conn, 3, "status", "draft")
    ids = _matching_ids(
        conn, {"key": "status", "operator": "in", "value": ["published", "archived"]}
    )
    assert ids == [1, 2]


def test_nin_requires_present_value_not_in_set():
    conn = _make_conn()
    _insert(conn, 1, "status", "published")
    _insert(conn, 2, "status", "draft")
    conn.execute("INSERT INTO documents (id) VALUES (3)")  # no status key -- must not match nin
    conn.commit()
    assert _matching_ids(conn, {"key": "status", "operator": "nin", "value": ["published"]}) == [2]


def test_all_requires_every_membership_value_present():
    conn = _make_conn()
    _insert(conn, 1, "topics", "python", ordinal=0)
    _insert(conn, 1, "topics", "sql", ordinal=1)
    _insert(conn, 2, "topics", "python", ordinal=0)
    assert _matching_ids(
        conn, {"key": "topics", "operator": "all", "value": ["python", "sql"]}
    ) == [1]


def test_exists_true_requires_key_present():
    conn = _make_conn()
    _insert(conn, 1, "status", "published")
    conn.execute("INSERT INTO documents (id) VALUES (2)")
    conn.commit()
    assert _matching_ids(conn, {"key": "status", "operator": "exists", "value": True}) == [1]


def test_exists_false_matches_key_absent():
    conn = _make_conn()
    _insert(conn, 1, "status", "published")
    conn.execute("INSERT INTO documents (id) VALUES (2)")
    conn.commit()
    assert _matching_ids(conn, {"key": "status", "operator": "exists", "value": False}) == [2]


def test_boolean_values_round_trip_through_compile():
    conn = _make_conn()
    _insert(conn, 1, "active", True)
    _insert(conn, 2, "active", False)
    assert _matching_ids(conn, {"key": "active", "operator": "eq", "value": True}) == [1]


def test_and_group_requires_all_operands():
    conn = _make_conn()
    _insert(conn, 1, "status", "published", ordinal=0)
    _insert(conn, 1, "priority", 5, ordinal=0)
    _insert(conn, 2, "status", "published", ordinal=0)
    _insert(conn, 2, "priority", 1, ordinal=0)
    ids = _matching_ids(
        conn,
        {
            "operator": "and",
            "operands": [
                {"key": "status", "operator": "eq", "value": "published"},
                {"key": "priority", "operator": "gt", "value": 2},
            ],
        },
    )
    assert ids == [1]


def test_or_group_requires_any_operand():
    conn = _make_conn()
    _insert(conn, 1, "status", "published")
    _insert(conn, 2, "status", "draft")
    ids = _matching_ids(
        conn,
        {
            "operator": "or",
            "operands": [
                {"key": "status", "operator": "eq", "value": "published"},
                {"key": "status", "operator": "eq", "value": "draft"},
            ],
        },
    )
    assert ids == [1, 2]


def test_not_negates_inner_filter():
    conn = _make_conn()
    _insert(conn, 1, "status", "published")
    _insert(conn, 2, "status", "draft")
    ids = _matching_ids(
        conn, {"operator": "not", "operand": {"key": "status", "operator": "eq", "value": "draft"}}
    )
    # doc 1 matches (status != draft); the compiled SQL is a WHERE clause over
    # `documents`, so this only proves the NOT/EXISTS logic, not full-corpus semantics.
    assert 1 in ids
    assert 2 not in ids


def test_params_never_interpolate_sql_special_characters():
    conn = _make_conn()
    _insert(conn, 1, "note", "a'; DROP TABLE documents; --")
    ids = _matching_ids(
        conn, {"key": "note", "operator": "eq", "value": "a'; DROP TABLE documents; --"}
    )
    assert ids == [1]
    # If interpolation (not binding) were happening, this second query would
    # now fail with "no such table: documents".
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
