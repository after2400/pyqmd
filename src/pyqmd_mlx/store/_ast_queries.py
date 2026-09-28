"""Per-language tree-sitter query source and compiled-query caching for
_ast.py. Split out of _ast.py so the query source (long S-expression
strings) doesn't crowd the grammar-loading/status logic."""

from ._ast import SupportedLanguage

LANGUAGE_QUERIES: dict[SupportedLanguage, str] = {
    "typescript": """
        (export_statement) @export
        (class_declaration) @class
        (function_declaration) @func
        (method_definition) @method
        (interface_declaration) @iface
        (type_alias_declaration) @type
        (enum_declaration) @enum
        (import_statement) @import
        (lexical_declaration (variable_declarator value: (arrow_function))) @func
        (lexical_declaration (variable_declarator value: (function_expression))) @func
    """,
    "tsx": """
        (export_statement) @export
        (class_declaration) @class
        (function_declaration) @func
        (method_definition) @method
        (interface_declaration) @iface
        (type_alias_declaration) @type
        (enum_declaration) @enum
        (import_statement) @import
        (lexical_declaration (variable_declarator value: (arrow_function))) @func
        (lexical_declaration (variable_declarator value: (function_expression))) @func
    """,
    "javascript": """
        (export_statement) @export
        (class_declaration) @class
        (function_declaration) @func
        (method_definition) @method
        (import_statement) @import
        (lexical_declaration (variable_declarator value: (arrow_function))) @func
        (lexical_declaration (variable_declarator value: (function_expression))) @func
    """,
    "python": """
        (class_definition) @class
        (function_definition) @func
        (decorated_definition) @decorated
        (import_statement) @import
        (import_from_statement) @import
    """,
    "go": """
        (type_declaration) @type
        (function_declaration) @func
        (method_declaration) @method
        (import_declaration) @import
    """,
    "rust": """
        (struct_item) @struct
        (impl_item) @impl
        (function_item) @func
        (trait_item) @trait
        (enum_item) @enum
        (use_declaration) @import
        (type_item) @type
        (mod_item) @mod
    """,
}

# Aligned with the markdown BREAK_PATTERNS scale in _chunking.py (h1=100,
# h2=90, ...) so an AST break point competes fairly with regex break
# points in find_best_cutoff(). Matches Node's SCORE_MAP exactly.
SCORE_MAP: dict[str, int] = {
    "class": 100,
    "iface": 100,
    "struct": 100,
    "trait": 100,
    "impl": 100,
    "mod": 100,
    "export": 90,
    "func": 90,
    "method": 90,
    "decorated": 90,
    "type": 80,
    "enum": 80,
    "import": 60,
}

_QUERY_CACHE: dict[SupportedLanguage, object] = {}


def get_query(language: SupportedLanguage, grammar: object) -> object:
    """Get or compile the cached tree_sitter.Query for `language`."""
    if language not in _QUERY_CACHE:
        from tree_sitter import Query

        _QUERY_CACHE[language] = Query(grammar, LANGUAGE_QUERIES[language])
    return _QUERY_CACHE[language]
