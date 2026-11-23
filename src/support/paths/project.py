"""Locate retained paper artifacts without changing their sealed logical identities.

PATH_MAP.json records the approved file moves and byte-identical deduplication.
This resolver never searches for replacement results or falls back to raw copies.
Reader forecasts use explicit local paths; external archives are not required.
"""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _mapping():
    path = ROOT / 'results/provenance/PATH_MAP.json'
    with path.open(encoding='utf-8') as stream:
        record = json.load(stream)
    if record['version'] != 1:
        raise ValueError('Unsupported paper path map')
    for old, new in list(record['files'].items()) + list(record['directories'].items()):
        for relative in (old, new):
            candidate = Path(relative)
            if candidate.is_absolute() or '..' in candidate.parts or ':' in relative:
                raise ValueError(f'Unsafe paper path mapping: {relative}')
    return record


def project_file(value, root=ROOT):
    """Resolve a project file; leave external paths and file-like objects alone."""
    if not isinstance(value, (str, Path)):
        return value
    path, root = Path(value), Path(root)
    if path.is_absolute():
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            return value
    else:
        relative = path.as_posix()
        if '..' in path.parts:
            return value
    record = _mapping()
    target = record['files'].get(relative)
    if target is None:
        for old, new in sorted(record['directories'].items(), key=lambda item: -len(item[0])):
            if relative == old or relative.startswith(old + '/'):
                target = new + relative[len(old):]
                break
    if target is None:
        return value
    # A directory may have been translated before its filename was appended.
    target = record['files'].get(target, target)
    resolved = root / target
    if not resolved.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Paper path escapes project root: {value}')
    return resolved
