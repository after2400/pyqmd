"""Document-list output formatting for `multi-get`: json/csv/files/md/xml/cli.
Ported from formatter.ts's documentsTo* functions (formatter.ts:235-309) for
json/csv/files/md/xml. `cli` format is NOT one of those -- Node's multi-get
command has its own dedicated inline banner renderer (multiGet() in
src/cli/qmd.ts, the `else` branch after the xml case), a different code path
from documentsToMarkdown/formatter.ts's `md` format. Porting `cli` as a
markdown-formatter fallback was a real bug: it produced a different line
count than Node's actual `qmd multi-get` output.

`get` (single-document fetch) does NOT use this module -- it always prints
plain text, matching the original CLI (--format only applies to search,
query, and multi-get).
"""

import json as _json

from pyqmd_mlx.cli._output_search import escape_csv, escape_xml
from pyqmd_mlx.cli._types import DocumentEntry


def documents_to_json(entries: list[DocumentEntry]) -> str:
    output = []
    for e in entries:
        item: dict = {"file": e.display_path}
        if e.docid:
            item["docid"] = f"#{e.docid}"
        item["title"] = e.title
        if e.context:
            item["context"] = e.context
        if e.skipped:
            item["skipped"] = True
            item["reason"] = e.skip_reason
        else:
            item["body"] = e.body
        output.append(item)
    return _json.dumps(output, indent=2)


def documents_to_csv(entries: list[DocumentEntry]) -> str:
    header = "docid,file,title,context,skipped,body"
    rows = [header]
    for e in entries:
        rows.append(
            ",".join(
                [
                    escape_csv(f"#{e.docid}" if e.docid else ""),
                    escape_csv(e.display_path),
                    escape_csv(e.title),
                    escape_csv(e.context or ""),
                    "true" if e.skipped else "false",
                    escape_csv(e.skip_reason or "" if e.skipped else e.body),
                ]
            )
        )
    return "\n".join(rows)


def documents_to_files(entries: list[DocumentEntry]) -> str:
    lines = []
    for e in entries:
        docid_prefix = f"#{e.docid}," if e.docid else ""
        ctx = f',"{e.context.replace(chr(34), chr(34) * 2)}"' if e.context else ""
        status = ",[SKIPPED]" if e.skipped else ""
        lines.append(f"{docid_prefix}{e.display_path}{ctx}{status}")
    return "\n".join(lines)


def documents_to_markdown(entries: list[DocumentEntry]) -> str:
    blocks = []
    for e in entries:
        md = f"## {e.display_path}\n\n"
        if e.docid:
            md += f"**docid:** `#{e.docid}`\n\n"
        if e.title and e.title != e.display_path:
            md += f"**Title:** {e.title}\n\n"
        if e.context:
            md += f"**Context:** {e.context}\n\n"
        if e.skipped:
            md += f"> {e.skip_reason}\n"
        else:
            md += "```\n" + e.body + "\n```\n"
        blocks.append(md)
    return "\n".join(blocks)


def documents_to_xml(entries: list[DocumentEntry]) -> str:
    items = []
    for e in entries:
        docid_attr = f' docid="#{e.docid}"' if e.docid else ""
        xml = f"  <document{docid_attr}>\n"
        xml += f"    <file>{escape_xml(e.display_path)}</file>\n"
        xml += f"    <title>{escape_xml(e.title)}</title>\n"
        if e.context:
            xml += f"    <context>{escape_xml(e.context)}</context>\n"
        if e.skipped:
            xml += "    <skipped>true</skipped>\n"
            xml += f"    <reason>{escape_xml(e.skip_reason or '')}</reason>\n"
        else:
            xml += f"    <body>{escape_xml(e.body)}</body>\n"
        xml += "  </document>"
        items.append(xml)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<documents>\n'
        + "\n".join(items)
        + "\n</documents>"
    )


def documents_to_cli(entries: list[DocumentEntry]) -> str:
    """Node's dedicated CLI-format renderer for multi-get: an `=`-bordered
    "File: <path>  #<docid>" banner per document, not the markdown format.
    Reconstructed line-for-line from src/cli/qmd.ts's multiGet() CLI branch
    (console.log calls, each contributing `text + "\\n"`); the trailing
    newline is stripped since the caller passes this to typer.echo(), which
    adds its own -- matching Node's total byte output exactly, including
    the multi_get_two_documents parity scenario's line count."""
    bar = "=" * 60
    blocks = []
    for e in entries:
        id_suffix = f"  #{e.docid}" if e.docid else ""
        block = f"\n{bar}\nFile: {e.display_path}{id_suffix}\n{bar}\n\n"
        if e.skipped:
            block += f"[SKIPPED: {e.skip_reason}]\n"
        else:
            if e.context:
                block += f"Folder Context: {e.context}\n---\n\n"
            block += f"{e.body}\n"
        blocks.append(block)
    result = "".join(blocks)
    return result[:-1] if result.endswith("\n") else result


def format_documents(entries: list[DocumentEntry], format: str) -> str:
    if format == "json":
        return documents_to_json(entries)
    if format == "csv":
        return documents_to_csv(entries)
    if format == "files":
        return documents_to_files(entries)
    if format == "md":
        return documents_to_markdown(entries)
    if format == "xml":
        return documents_to_xml(entries)
    if format == "cli":
        return documents_to_cli(entries)
    raise ValueError(f"Unknown format: {format}")
