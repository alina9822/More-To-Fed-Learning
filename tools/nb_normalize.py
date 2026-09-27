#!/usr/bin/env python3
"""Git clean filter: normalize .ipynb on the way into git so diffs stay readable.

Reads a notebook on stdin, writes the normalized notebook to stdout.
Outputs are preserved; only per-run and per-tool noise is removed.

Enable with:
    git config filter.nbnormalize.clean "python3 tools/nb_normalize.py"

Stdlib only, because the system Python is PEP 668 managed and cannot pip install.
"""
import json
import sys

# image/* payloads are single base64 blobs; splitting them helps nothing.
TEXTUAL_MIME_PREFIXES = ("text/", "application/javascript", "application/json")


def as_lines(value):
    """nbformat stores multiline strings as a list of lines, which diffs per line."""
    if isinstance(value, list):
        value = "".join(value)
    if not isinstance(value, str):
        return value
    return value.splitlines(keepends=True)


def normalize_output(output):
    output.pop("execution_count", None)
    if "text" in output:
        output["text"] = as_lines(output["text"])
    data = output.get("data")
    if isinstance(data, dict):
        for mime, payload in data.items():
            if mime.startswith(TEXTUAL_MIME_PREFIXES):
                data[mime] = as_lines(payload)
    return output


def normalize(nb):
    meta = nb.get("metadata", {})
    # All of this churns purely from which machine/tool last opened the notebook:
    # VS Code writes ".venv (3.12.3)", Colab writes "Python 3". Both tools ignore
    # the stored value and use whichever kernel the user actually selected.
    meta.get("language_info", {}).pop("version", None)
    if "kernelspec" in meta:
        meta["kernelspec"] = {
            "display_name": "Python 3",
            "language": meta["kernelspec"].get("language", "python"),
            "name": "python3",
        }

    for cell in nb.get("cells", []):
        cell["source"] = as_lines(cell.get("source", ""))
        cell_meta = cell.get("metadata", {})
        cell_meta.pop("outputId", None)  # Colab regenerates this every execution
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            for output in cell.get("outputs", []):
                normalize_output(output)
        else:
            cell.pop("execution_count", None)
            cell.pop("outputs", None)
    return nb


def main():
    raw = sys.stdin.read()
    try:
        text = json.dumps(
            normalize(json.loads(raw)),
            indent=1,
            sort_keys=True,
            ensure_ascii=False,
        )
    except Exception:
        # Never let this filter be the reason a commit loses or mangles content.
        sys.stdout.write(raw)
        return
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
