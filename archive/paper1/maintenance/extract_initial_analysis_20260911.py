"""One-time mechanical extraction; retained as a migration recipe, not an entrypoint.

Run only against the recorded pre-refactor source. Numerical function bodies are
preserved; path-owning orchestration is changed separately and regression-tested.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'scripts/dsol_paper1/analyze_standard_initial_results_v1.py'
PACKAGE = ROOT/'AlphaBrain/research/dsol'


def main():
    text = SOURCE.read_text(); tree = ast.parse(text)
    nodes = {n.name: ast.get_source_segment(text,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    contract = next(ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONTRACT' for t in n.targets))
    outputs = {
        'data/artifacts.py': ('from pathlib import Path\nimport hashlib\nimport json\nimport os\n', ['read','sha','write']),
        'metrics/view_rules.py': ('import numpy as np\n', ['metric_choices']),
        'selectors/specification.py': ('', []),
        'selectors/ridge.py': ('import numpy as np\nfrom .specification import CONTRACT\n', ['pca_fit','feature_matrix','ridge_predict','fit_ranker']),
        'analysis/statistics.py': ('import numpy as np\n', ['hierarchy','paired_interval']),
        'analysis/initial_study.py': ('import numpy as np\nfrom scipy.stats import spearmanr\nfrom ..data.artifacts import write\nfrom ..metrics.view_rules import metric_choices\nfrom ..selectors.specification import CONTRACT\nfrom ..selectors.ridge import fit_ranker\nfrom .statistics import hierarchy, paired_interval\nMODELS = ["canonical", "broad"]\n', ['summarize']),
        'data/initial_matrix.py': ('from pathlib import Path\nimport json\nimport hashlib\nfrom concurrent.futures import ThreadPoolExecutor\nimport numpy as np\nfrom .artifacts import read, sha, write\nMODELS = ["canonical", "broad"]\n', ['audit_and_load']),
        'selectors/visual_features.py': ('import numpy as np\nfrom ..data.artifacts import sha\nfrom .specification import CONTRACT\n', ['encode_assets']),
    }
    receipt = {'original_source':str(SOURCE.relative_to(ROOT)), 'original_sha256':hashlib.sha256(text.encode()).hexdigest(), 'moves':[]}
    for relative,(imports,names) in outputs.items():
        p=PACKAGE/relative
        if p.exists():raise RuntimeError('Target exists: '+str(p))
        p.parent.mkdir(parents=True,exist_ok=True)
        init=p.parent/'__init__.py'
        if not init.exists():init.write_text('"""Paper 1 reusable '+p.parent.name+' modules."""\n')
        body='\n\n\n'.join(nodes[n] for n in names)
        if relative=='selectors/specification.py':body=contract
        if relative=='data/initial_matrix.py':body=body.replace('def audit_and_load(out):','def audit_and_load(root, out):').replace('ROOT','root')
        if relative=='selectors/visual_features.py':body=body.replace('def encode_assets(out):','def encode_assets(root, out, weights):').replace('ROOT','root').replace('WEIGHTS','weights')
        payload='"""Extracted from the audited INITIAL32 analysis; preserve scientific semantics."""\n'+imports+'\n\n'+body+'\n'
        p.write_text(payload);receipt['moves'].append({'module':str(p.relative_to(ROOT)),'functions':names,'sha256':hashlib.sha256(payload.encode()).hexdigest()})
    archive=ROOT/'archive/paper1/maintenance/baseline-20260911'
    archive.mkdir(parents=True,exist_ok=True)
    (archive/'analyze_standard_initial_results_v1.py').write_text(text)
    (ROOT/'archive/paper1/maintenance/extraction-20260911.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
