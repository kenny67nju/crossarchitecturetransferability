"""Check that every figure/table/algorithm is cited outside all floats.

This is a structural check; whether the surrounding prose explains each item
must still be reviewed manually. Includes input files and appendix prose.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FLOAT = re.compile(r"\\begin\{(figure\*?|table\*?|algorithm)\}(.*?)\\end\{\1\}", re.S)
REF = re.compile(r"\\(?:ref|cref|Cref|autoref)\{([^}]+)\}")


def expand(path, stack=()):
    path = path.resolve()
    if path in stack:
        raise ValueError(f"Recursive input: {path}")
    source = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text())

    def include(match):
        target = PAPER / match.group(1)
        if not target.suffix:
            target = target.with_suffix(".tex")
        return expand(target, (*stack, path))

    return re.sub(r"\\input\{([^}]+)\}", include, source)


def audit(filename):
    source = expand(PAPER / filename)
    floats = list(FLOAT.finditer(source))
    # Exclude every float, not just the float being checked: references in
    # another caption must not count as explanatory body references.
    body = FLOAT.sub(lambda m: "\n" * m.group(0).count("\n"), source)
    references = {label.strip() for match in REF.finditer(body)
                  for label in match.group(1).split(",")}
    errors = []
    counts = {}
    for item in floats:
        kind = item.group(1)
        counts[kind] = counts.get(kind, 0) + 1
        labels = re.findall(r"\\label\{([^}]+)\}", item.group(2))
        if len(labels) != 1:
            errors.append(f"{kind}: expected one label, got {labels}")
            continue
        label = labels[0]
        if label not in references:
            errors.append(f"No body reference: {label}")
        print(f"{filename}: {kind} {label}: {'PASS' if label in references else 'FAIL'}")
    assert not errors, "\n".join(errors)
    print(f"{filename}: all {len(floats)} floats cited outside floats; {counts}")


if __name__ == "__main__":
    for filename in ("main_v18.tex", "main_v18_zh.tex"):
        audit(filename)
