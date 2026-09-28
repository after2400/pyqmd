"""Synthetic fixture for AST-aware chunking parity coverage. Not real
code -- exists purely to exceed CHUNK_SIZE_CHARS with multiple
function definitions, so embed --chunk-strategy auto has boundaries to
align to."""

import json


def alpha_handler(request):
    """Handle an alpha request."""
    payload = {"status": "ok", "kind": "alpha"}
    return json.dumps(payload)


def bravo_handler(request):
    """Handle a bravo request."""
    payload = {"status": "ok", "kind": "bravo"}
    return json.dumps(payload)


def charlie_handler(request):
    """Handle a charlie request."""
    payload = {"status": "ok", "kind": "charlie"}
    return json.dumps(payload)


def delta_handler(request):
    """Handle a delta request."""
    payload = {"status": "ok", "kind": "delta"}
    return json.dumps(payload)


def echo_handler(request):
    """Handle an echo request."""
    payload = {"status": "ok", "kind": "echo"}
    return json.dumps(payload)


def foxtrot_handler(request):
    """Handle a foxtrot request."""
    payload = {"status": "ok", "kind": "foxtrot"}
    return json.dumps(payload)


def golf_handler(request):
    """Handle a golf request."""
    payload = {"status": "ok", "kind": "golf"}
    return json.dumps(payload)


def hotel_handler(request):
    """Handle a hotel request."""
    payload = {"status": "ok", "kind": "hotel"}
    return json.dumps(payload)


def india_handler(request):
    """Handle an india request."""
    payload = {"status": "ok", "kind": "india"}
    return json.dumps(payload)


def juliet_handler(request):
    """Handle a juliet request."""
    payload = {"status": "ok", "kind": "juliet"}
    return json.dumps(payload)


def kilo_handler(request):
    """Handle a kilo request."""
    payload = {"status": "ok", "kind": "kilo"}
    return json.dumps(payload)


def lima_handler(request):
    """Handle a lima request."""
    payload = {"status": "ok", "kind": "lima"}
    return json.dumps(payload)


def mike_handler(request):
    """Handle a mike request."""
    payload = {"status": "ok", "kind": "mike"}
    return json.dumps(payload)


def november_handler(request):
    """Handle a november request."""
    payload = {"status": "ok", "kind": "november"}
    return json.dumps(payload)


def oscar_handler(request):
    """Handle an oscar request."""
    payload = {"status": "ok", "kind": "oscar"}
    return json.dumps(payload)


def papa_handler(request):
    """Handle a papa request."""
    payload = {"status": "ok", "kind": "papa"}
    return json.dumps(payload)


def quebec_handler(request):
    """Handle a quebec request."""
    payload = {"status": "ok", "kind": "quebec"}
    return json.dumps(payload)


def romeo_handler(request):
    """Handle a romeo request."""
    payload = {"status": "ok", "kind": "romeo"}
    return json.dumps(payload)


def sierra_handler(request):
    """Handle a sierra request."""
    payload = {"status": "ok", "kind": "sierra"}
    return json.dumps(payload)


class RequestRouter:
    """Routes requests to the handler matching their kind."""

    def __init__(self):
        self.handlers = {
            "alpha": alpha_handler,
            "bravo": bravo_handler,
            "charlie": charlie_handler,
            "delta": delta_handler,
            "echo": echo_handler,
            "foxtrot": foxtrot_handler,
            "golf": golf_handler,
            "hotel": hotel_handler,
            "india": india_handler,
            "juliet": juliet_handler,
            "kilo": kilo_handler,
            "lima": lima_handler,
            "mike": mike_handler,
            "november": november_handler,
            "oscar": oscar_handler,
            "papa": papa_handler,
            "quebec": quebec_handler,
            "romeo": romeo_handler,
            "sierra": sierra_handler,
        }

    def route(self, kind, request):
        handler = self.handlers.get(kind)
        if handler is None:
            raise ValueError(f"no handler for kind: {kind}")
        return handler(request)

    def known_kinds(self):
        return sorted(self.handlers)
