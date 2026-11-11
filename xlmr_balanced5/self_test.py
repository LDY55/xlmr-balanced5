from pathlib import Path
import hashlib,json
from detector import XLMRDetector
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
for name,expected in manifest['files'].items():
    with (root/name).open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual==expected, 'Checksum mismatch: '+name
d=XLMRDetector(root/'model',device='cpu',threads=8)
for r in json.loads((root/'self_test_cases.json').read_text(encoding='utf-8')):
    result=d.check(r['text'],r['lang'],8)
    assert result['decision']==r['decision'],r['lang']
    assert abs(result['logit']-r['logit'])<1e-3,(r['lang'],result['logit'],r['logit'])
assert d.check('', 'ru')['decision']=='abstain'
print('Checksums and CPU inference passed for all five languages.')
