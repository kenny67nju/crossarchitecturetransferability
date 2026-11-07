"""Read-only acceptance checks for the bilingual camera-ready PDF exports.

Requires pypdf and Poppler's pdffonts. This checks document structure, selected
audited numbers and link integrity; it is not a rerun of all historical studies.
"""
from pathlib import Path
import csv
import json
import re
import subprocess

from pypdf import PdfReader
from build_camera_ready_pdfs import destination

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
P0 = ROOT / "results/p0_vgg_pair_20261104/p0_aggregate_v2"


def text(page):
    return " ".join((page.extract_text() or "").split())


def annotations(page):
    value = page.get("/Annots", [])
    return value.get_object() if hasattr(value, "get_object") else value


def main():
    with (P0 / "single_preprocess_4096_common22.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 22 and len({r["task"] for r in rows}) == 10
    assert all(r["task"] != "stl10" for r in rows)
    pool_values = {}
    for pool, n in [("MV", 5), ("VTAB", 7), ("EX", 10)]:
        subset = [r for r in rows if r["pool"] == pool]
        assert len(subset) == n
        rm = sum(float(r["rankme_regret"]) for r in subset) / n
        ssr = sum(float(r["ssr_regret"]) for r in subset) / n
        pool_values[pool] = dict(tasks=n, rankme=rm, ssr=ssr)
        for filename in ["main_v18.tex", "main_v18_zh.tex"]:
            source = (PAPER / filename).read_text()
            assert f"${rm:.4f}$ & ${ssr:.4f}$ & ${ssr-rm:.4f}$" in source
    counts = {}
    with (P0 / "single_preprocess_4096_pool132_scores.csv").open() as f:
        for row in csv.DictReader(f):
            counts.setdefault(row["arch"], set()).add(row["variant"])
    multi = {k: len(v) for k, v in counts.items() if len(v) >= 2}
    assert len(counts) == 48 and len(multi) == 35
    assert sum(multi.values()) == 119 and (min(multi.values()), max(multi.values())) == (2, 11)
    for filename in ["main_v18.tex", "main_v18_zh.tex"]:
        source = (PAPER / filename).read_text()
        for key, count in multi.items():
            assert "\\texttt{" + key.replace("_", "\\_") + "} & $" + str(count) + "$" in source
        assert "\\label{page:content_end}" in source

    pareto = (ROOT / "code/figures_pareto.py").read_text()
    assert "build_p0_camera_assets" in pareto
    report = {"per_pool": pool_values, "pdfs": {}}
    for lang in ["en", "zh"]:
        expected_title=("Width Confounding in Label-Free Cross-Architecture Model Selection: Diagnosis and Subspace Correction"
                        if lang=="en" else "无标签跨架构模型选择中的宽度混淆：诊断与子空间校正")
        paths = [PAPER / f"Paper06_camera_ready_{lang}.pdf",
                 PAPER / f"Paper06_supplement_{lang}.pdf",
                 PAPER / f"Paper06_camera_ready_combined_{lang}.pdf"]
        main_pdf, supplement, combined = [PdfReader(p) for p in paths]
        assert paths[0].read_bytes() == paths[2].read_bytes()
        marker = "Supplementary Material" if lang == "en" else "补充材料"
        boundaries = [i for i, p in enumerate(main_pdf.pages)
                      if marker in (p.extract_text() or "")[:160]]
        assert len(boundaries) == 1, boundaries
        split = boundaries[0]
        checklist = "NeurIPS Paper Checklist" if lang == "en" else "NeurIPS 论文核查表"
        checklist_pages = [i for i,p in enumerate(main_pdf.pages) if checklist in text(p)[:200]]
        assert len(checklist_pages)==1 and checklist_pages[0]>split, checklist_pages
        appendix_end=checklist_pages[0]
        assert appendix_end - split == len(supplement.pages)
        for a, b in zip(main_pdf.pages[split:appendix_end], supplement.pages):
            assert a.extract_text() == b.extract_text()
            assert len(annotations(a)) == len(annotations(b))
        end = "as open problems." if lang == "en" else "仍是开放问题。"
        end_pages = [i + 1 for i, p in enumerate(main_pdf.pages[:split])
                     if end.replace(" ", "") in text(p).replace(" ", "")]
        assert len(end_pages) == 1 and end_pages[0] <= 10, end_pages
        all_text = "\n".join(text(p) for p in main_pdf.pages)
        conclusion = "8 Conclusion" if lang == "en" else "8 结论"
        ack = "Acknowledgments and Disclosure of Funding"
        ack_pages = [i + 1 for i, p in enumerate(main_pdf.pages) if ack in text(p)]
        assert ack_pages == [end_pages[0] + 1], (end_pages, ack_pages)
        refs = "References" if lang == "en" else "参考文献"
        checklist = "NeurIPS Paper Checklist" if lang == "en" else "NeurIPS 论文核查表"
        assert all_text.index(conclusion) < all_text.index(ack) < all_text.index(refs) < all_text.index(marker) < all_text.index(checklist)
        compact = re.sub(r"\s+", "", all_text)
        assert all(x in compact for x in ["B26023", "JYB2025XDXM118", "0.133", "0.069", "0.00194", "0.0645"])
        body_compact=re.sub(r"\s+","","\n".join(text(p) for p in main_pdf.pages[:split]))
        assert all(x not in body_compact for x in ["0.319", "0.173", "0.1043", "0.00118", "0.00254"])
        assert all(x in body_compact for x in ["0.058","0.101","0.305","0.512","37.4/36.4"])
        assert ("no competing interests" if lang == "en" else "不存在利益冲突") in all_text
        for path, reader in zip(paths, [main_pdf, supplement, combined]):
            title=expected_title + ((" - Supplementary Material" if lang=="en" else " - 补充材料") if path==paths[1] else "")
            assert reader.metadata.title==title,(path,reader.metadata.title)
            assert reader.metadata.author=="Qin Liu, Chen Zhong, Fengshan Zhao",(path,reader.metadata.author)
            assert path.stat().st_size < 50_000_000
            font_listing = subprocess.check_output(["pdffonts", str(path)], text=True)
            assert "Type 3" not in font_listing
            for row in font_listing.splitlines()[2:]:
                assert re.search(r"\byes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$", row), row
            for page in reader.pages:
                assert tuple(float(v) for v in page.mediabox) == (0, 0, 612, 792)
                for ref in annotations(page):
                    annotation = ref.get_object()
                    action = annotation.get("/A")
                    if action:
                        action = action.get_object()
                        if action.get("/S") == "/GoToR":
                            assert path == paths[1], 'Complete camera-ready must not depend on another PDF'
                            target = PdfReader(path.parent / str(action["/F"]))
                            assert 0 <= int(action["/D"][0]) < len(target.pages)
                        elif action.get("/S") == "/GoTo":
                            i, _ = destination(reader, action["/D"])
                            assert 0 <= i < len(reader.pages)
                    elif annotation.get("/Dest") is not None:
                        i, _ = destination(reader, annotation["/Dest"])
                        assert 0 <= i < len(reader.pages)
        report["pdfs"][lang] = dict(content_pages=end_pages[0], disclosures_start_page=ack_pages[0],
            appendices_start_page=split+1, checklist_start_page=appendix_end+1, supplement_pages=len(supplement.pages),
            camera_ready_pages=len(main_pdf.pages), delivery_mode='complete_with_appendices',
            combined_pages=len(combined.pages),metadata_verified=True)
    report['passed'] = True
    report['visual_review'] = 'Not performed by this script; see docs/RELEASE_VERIFICATION.md for manual checks.'
    (ROOT / 'results/camera_ready_p0_20261104/pdf_acceptance.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print("PDF acceptance checks passed (visual review is separate).")


if __name__ == "__main__":
    main()
