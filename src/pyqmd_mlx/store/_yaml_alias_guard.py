"""pyyaml has no built-in equivalent to the JS `yaml` package's
`maxAliasCount` option (checked directly against pyyaml==6.0.3) -- a
document with a small byte count can still expand exponentially through
nested YAML aliases ("billion laughs"). This loader counts every alias
resolved during composition and raises once a caller-supplied limit is
exceeded, closing that gap without adding a dependency."""

import yaml
from yaml.events import AliasEvent


class _AliasLimitedSafeLoader(yaml.SafeLoader):
    def __init__(self, stream, max_alias_count: int):
        super().__init__(stream)
        self._alias_count = 0
        self._max_alias_count = max_alias_count

    def compose_node(self, parent, index):
        if self.check_event(AliasEvent):
            self._alias_count += 1
            if self._alias_count > self._max_alias_count:
                raise yaml.YAMLError(f"too many YAML aliases (max {self._max_alias_count})")
        return super().compose_node(parent, index)


def safe_load_with_alias_limit(text: str, max_alias_count: int):
    """Like `yaml.safe_load`, but raises `yaml.YAMLError` if the document
    resolves more than `max_alias_count` YAML aliases during parsing."""
    loader = _AliasLimitedSafeLoader(text, max_alias_count)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()
