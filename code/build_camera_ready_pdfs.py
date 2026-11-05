"""Compile complete bilingual camera-ready PDFs, including appendices.

Requires XeLaTeX, BibTeX and pypdf. The canonical TeX contains all content:
main paper, bibliography, appendices, then the checklist.
The default Paper06_camera_ready_<language>.pdf contains all pages and internal
links. The combined filename is a compatibility alias. A separate supplement
is retained only as an optional extract, not as a required submission companion.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject, BooleanObject, DictionaryObject, NameObject,
    NumberObject, TextStringObject,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUTPUT = PAPER


def run(command: list[str], env: dict[str, str], log: Path, cwd: Path = PAPER) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed; see {log}\n{result.stdout[-3000:]}")


def destination(reader: PdfReader, value):
    """Resolve a source destination to its zero-based page and view parameters."""
    if isinstance(value, str):
        dest = reader.named_destinations.get(str(value).lstrip("/"))
        if dest is None:
            dest = reader.named_destinations.get(str(value))
        if dest is None:
            raise ValueError(f"Unresolved destination: {value}")
        array = dest.dest_array
        page = reader.get_destination_page_number(dest)
    else:
        array = value.get_object()
        ref = array[0]
        page = next((i for i, p in enumerate(reader.pages)
                     if p.indirect_reference == ref), None)
        if page is None and isinstance(ref, int):
            page = int(ref)
    if page is None:
        raise ValueError(f"Unresolved destination page: {value}")
    return page, list(array[1:])


def export_part(reader: PdfReader, start: int, end: int, split: int,
                path: Path, sibling: str) -> int:
    writer = PdfWriter()
    for page in reader.pages[start:end]:
        # Rebuild annotations below: default cloning drops links to omitted pages.
        saved_annotations = page.pop("/Annots", None)
        try:
            writer.add_page(page)
        finally:
            if saved_annotations is not None:
                page[NameObject("/Annots")] = saved_annotations
    suffix = "补充材料" if path.stem.endswith("_zh") else "Supplementary Material"
    writer.add_metadata({"/Title": reader.metadata.title + " - " + suffix,
                         "/Author": reader.metadata.author})
    remote_links = 0
    for source_i in range(start, end):
        old_page = reader.pages[source_i]
        new_page = writer.pages[source_i - start]
        old_annots = old_page.get("/Annots", ArrayObject()).get_object()
        for old_ref in old_annots:
            old = old_ref.get_object()
            new = DictionaryObject({
                key: value.clone(writer) for key, value in old.items()
                if key not in ("/A", "/Dest", "/P")
            })
            value = old.get("/Dest")
            action = old.get("/A")
            if action is not None:
                action = action.get_object()
                if action.get("/S") == "/GoTo":
                    value = action["/D"]
            if value is None:
                if action is not None:
                    new[NameObject("/A")] = action.clone(writer)
                writer.add_annotation(source_i - start, new)
                continue
            target, view = destination(reader, value)
            new.pop("/Dest", None)
            if start <= target < end:
                dest = ArrayObject([writer.pages[target - start].indirect_reference] + view)
                new[NameObject("/A")] = DictionaryObject({
                    NameObject("/S"): NameObject("/GoTo"),
                    NameObject("/D"): dest,
                })
            else:
                # The sibling is the complete camera-ready, so use its physical page.
                remote_target = target
                new[NameObject("/A")] = DictionaryObject({
                    NameObject("/S"): NameObject("/GoToR"),
                    NameObject("/F"): TextStringObject(sibling),
                    NameObject("/D"): ArrayObject([NumberObject(remote_target)] + view),
                    NameObject("/NewWindow"): BooleanObject(True),
                })
                remote_links += 1
            writer.add_annotation(source_i - start, new)
    writer.set_page_label(0, end - start - 1, style="/D",
                          prefix="S" if start == split else "", start=1)
    with path.open("wb") as stream:
        writer.write(stream)
    check = PdfReader(path)
    assert len(check.pages) == end - start
    for i, page in enumerate(check.pages):
        source_page = reader.pages[start + i]
        assert page.extract_text() == source_page.extract_text()
        assert len(page.get("/Annots", ArrayObject()).get_object()) == len(
            source_page.get("/Annots", ArrayObject()).get_object())
    # Check every internal link resolves inside this exported PDF.
    for page in check.pages:
        for ref in page.get("/Annots", ArrayObject()).get_object():
            annotation = ref.get_object()
            action = annotation.get("/A")
            if action is not None:
                action = action.get_object()
                if action.get("/S") == "/GoTo":
                    target, _ = destination(check, action["/D"])
                    assert 0 <= target < len(check.pages)
    return remote_links


def build(language: str) -> None:
    stem = "main_v18" if language == "en" else "main_v18_zh"
    marker = "Supplementary Material" if language == "en" else "补充材料"
    stage = Path(tempfile.mkdtemp(prefix=f"paper06-{language}-"))
    env = dict(os.environ)
    env["BIBINPUTS"] = str(PAPER) + os.pathsep + env.get("BIBINPUTS", "")
    latex = ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={stage}", stem + ".tex"]
    run(latex, env, stage / "pass1.txt")
    run(["bibtex", stem], env, stage / "bibtex.txt", cwd=stage)
    for index in range(2, 5):
        run(latex, env, stage / f"pass{index}.txt")
    log = (stage / (stem + ".log")).read_text(errors="replace")
    for error in ("There were undefined references", "There were undefined citations",
                  "multiply defined", "Missing character:"):
        if error in log:
            raise RuntimeError(f"{error}; see {stage / (stem + '.log')}")
    reader = PdfReader(stage / (stem + ".pdf"))
    indices = [i for i, p in enumerate(reader.pages)
               if marker in (p.extract_text() or "")[:160]]
    if len(indices) != 1:
        raise ValueError(f"Expected one supplement boundary, found {indices}")
    split = indices[0]
    checklist_marker = "NeurIPS Paper Checklist" if language == "en" else "NeurIPS 论文核查表"
    checklist_indices = [i for i,p in enumerate(reader.pages)
                         if checklist_marker in " ".join((p.extract_text() or "").split())[:200]]
    if len(checklist_indices) != 1 or checklist_indices[0] <= split:
        raise ValueError(f"Expected checklist after appendices: {checklist_indices}")
    appendix_end = checklist_indices[0]
    conclusion = "8 Conclusion" if language == "en" else "8 结论"
    conclusion_pages = [i + 1 for i, p in enumerate(reader.pages[:split])
                        if conclusion in " ".join((p.extract_text() or "").split())]
    if len(conclusion_pages) != 1 or conclusion_pages[0] > 10:
        raise ValueError(f"Cannot confirm <=10 content pages: {conclusion_pages}")
    # Check the END of the conclusion, not merely its heading page.
    aux = (stage / (stem + ".aux")).read_text()
    # Keep table identifiers aligned in the English paper and Chinese translation.
    for label, number in [("official_disco",7),("pool_regrets",8),("crosspool",9),("two_stage",10)]:
        key="tab:"+label+("_zh" if language=="zh" else "")
        match=re.search(r"\\newlabel\{"+re.escape(key)+r"\}\{\{(\d+)\}",aux)
        if match is None or int(match.group(1)) != number:
            raise ValueError(f"Wrong table numbering for {key}: {match}")
    ending = re.search(r"\\newlabel\{page:content_end\}\{\{[^}]*\}\{(\d+)\}", aux)
    if ending is None or int(ending.group(1)) > 10:
        raise ValueError("Conclusion ends after page 10 or has no end-page label")
    content_pages = int(ending.group(1))
    main_name = f"Paper06_camera_ready_{language}.pdf"
    supp_name = f"Paper06_supplement_{language}.pdf"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stage / (stem + ".pdf"), OUTPUT / main_name)
    links_supp = export_part(reader, split, appendix_end, split,
                             OUTPUT / supp_name, main_name)
    combined = OUTPUT / f"Paper06_camera_ready_combined_{language}.pdf"
    shutil.copy2(stage / (stem + ".pdf"), combined)
    assert (OUTPUT / main_name).read_bytes() == combined.read_bytes()
    print(f"{language}: content={content_pages} pages; complete camera-ready={len(reader.pages)} pages; "
          f"appendices={appendix_end-split} pages; checklist={len(reader.pages)-appendix_end} pages; optional extract links={links_supp}; build logs={stage}", flush=True)
    for line in log.splitlines():
        if "Overfull" in line:
            print(line, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("en", "zh", "both"), default="both")
    args = parser.parse_args()
    for language in (("en", "zh") if args.language == "both" else (args.language,)):
        build(language)
