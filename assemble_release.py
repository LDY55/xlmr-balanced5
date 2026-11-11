"""Assemble the verified balanced5 ZIP from GitHub release parts (Python 3.11)."""
from pathlib import Path
import argparse
import hashlib
import json

EXPECTED_SHA256 = '2087f6bdba4a7c4b5377d4d4ace94e04faefda89b516f9dc0b2285d4e8fe8d3c'
EXPECTED_NAME = 'xlmr_balanced5_release_20261111.zip'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def assemble(directory):
    spec = json.loads((directory / 'parts_manifest.json').read_text(encoding='utf-8'))
    assert spec['archive'] == EXPECTED_NAME and spec['sha256'] == EXPECTED_SHA256
    assert spec['bytes'] == 1130040554 and len(spec['parts']) == 17
    output = directory / EXPECTED_NAME
    if output.exists():
        if digest(output) != EXPECTED_SHA256:
            raise FileExistsError('Output already exists with a different checksum: ' + str(output))
        print('Archive already verified:', output)
        return
    for i, part in enumerate(spec['parts'], 1):
        assert part['name'] == EXPECTED_NAME + f'.part{i:03d}'
        path = directory / part['name']
        if not path.is_file() or path.stat().st_size != part['size']:
            raise ValueError('Missing or incomplete part: ' + str(path))
    temporary = output.with_suffix('.zip.assembling')
    full_hash = hashlib.sha256()
    with temporary.open('wb') as dest:
        for i, part in enumerate(spec['parts'], 1):
            part_hash = hashlib.sha256()
            with (directory / part['name']).open('rb') as source:
                while chunk := source.read(8 * 1024 * 1024):
                    dest.write(chunk)
                    part_hash.update(chunk)
                    full_hash.update(chunk)
            if part_hash.hexdigest() != part['sha256']:
                raise ValueError('Checksum mismatch: ' + part['name'])
            print(f'Verified part {i}/17', flush=True)
    assert temporary.stat().st_size == spec['bytes']
    if full_hash.hexdigest() != EXPECTED_SHA256:
        raise ValueError('Archive checksum mismatch')
    temporary.rename(output)
    print('Archive assembled and verified:', output)
    print('SHA-256:', EXPECTED_SHA256)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    assemble(parser.parse_args().directory.resolve())
