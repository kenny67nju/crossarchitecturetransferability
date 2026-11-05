"""Read-only coverage checks for dataset attribution and bilingual checklists."""
from pathlib import Path
import csv
import re

ROOT = Path(__file__).resolve().parents[1]

def main():
    with (ROOT / 'results/paper14_crossdomain.csv').open() as stream:
        tasks = {row['task'] for row in csv.DictReader(stream)}
    inventory = (ROOT / 'docs/DATASET_SOURCES.md').read_text()
    rows = re.findall(r'^\| `([^`]+)` \| ([^|]+) \|', inventory, re.M)
    mapping = {name: re.findall(r'`([^`]+)`', keys) for name, keys in rows}
    assert tasks == set(mapping), (tasks - set(mapping), set(mapping) - tasks)
    bib = (ROOT / 'paper/references.bib').read_text()
    keys = re.findall(r'@\w+\{([^,]+),', bib)
    assert len(keys) == len(set(keys)), 'Duplicate BibTeX key'
    required = {k for entries in mapping.values() for k in entries}
    required |= {'deng2009imagenet', 'zhai2019large'}
    assert required <= set(keys), required - set(keys)
    for filename in ['main_v18.tex', 'main_v18_zh.tex']:
        source = (ROOT / 'paper' / filename).read_text()
        cited = {key.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}', source)
                 for key in group.split(',')}
        assert required <= cited, (filename, required - cited)
        assert cited <= set(keys), (filename, cited - set(keys))
    answers = []
    for filename in ['checklist.tex', 'checklist_zh.tex']:
        source = (ROOT / 'paper' / filename).read_text()
        lines = [line for line in source.splitlines()
                 if re.search(r'\\item\[\]\s*(Answer:|回答：)', line)]
        values = [re.search(r'\\answer(Yes|No|NA)', line).group(1) for line in lines]
        assert len(values) == 16, (filename, len(values))
        assert values[3:6] == ['No', 'No', 'Yes']
        assert values[7] == 'No' and values[11:13] == ['No', 'No']
        answers.append(values)
        assert 'require repository release' not in source
        assert '仍需在仓库发布' not in source
        assert 'library licenses are documented' not in source
        assert '仓库记录了库许可证' not in source
    assert answers[0] == answers[1], 'Bilingual checklist answers differ'
    print(f'PASS: {len(tasks)} tasks; {len(required)} required source citations in both papers; 16 aligned checklist answers.')

if __name__ == '__main__':
    main()
