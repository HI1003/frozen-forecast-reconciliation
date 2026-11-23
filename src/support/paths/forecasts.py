"""Canonical frozen-forecast storage and explicit historical path translation.

Translation changes only the location, never the model or experiment identity.
Sealed manifests retain their logical paths and SHA-256 checks. No lookup in an
old directory is attempted if the canonical file is missing.
"""


from pathlib import Path
import re
from src.support.paths.project import project_file


def _component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Unsafe base artifact component: {value!r}")
    return value


def base_forecast_path(root, experiment_id, dataset, base):
    if not isinstance(experiment_id, str) or not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,63}', experiment_id):
        raise ValueError(f'Unsafe experiment ID: {experiment_id!r}')
    dataset, base = map(_component, (dataset, base))
    # Keep independent experimental caches distinct without extra directories.
    suffix = '' if experiment_id == 'main' else f'__experiment-{experiment_id}'
    return (Path(root) / 'results/raw_local/base_forecasts' / dataset
            / f'{dataset}_{base}{suffix}.npz')


def resolve_base_path(path):
    """Resolve only the exact historical experiment/base filename grammar.

    Unnamespaced smoke outputs are deliberately NOT mapped onto formal main.
    Unrelated paths (including paths in downloaded archives) pass through.
    """
    path = Path(path)
    if (path.parent.name != 'base_forecasts'
            or path.parent.parent.parent.name != 'experiments'
            or path.parent.parent.parent.parent.name != 'results'):
        return project_file(path)
    suffix = next((s for s in ('.meta.json', '.npz', '.publish.lock')
                   if path.name.endswith(s)), None)
    if suffix is None:
        return path
    stem = path.name[:-len(suffix)]
    dataset, sep, base = stem.partition('_')
    if not sep:
        raise ValueError(f'Invalid historical base filename: {path.name}')
    root = path.parent.parent.parent.parent.parent
    target = base_forecast_path(root, path.parent.parent.name, dataset, base)
    return project_file(target.with_suffix(suffix))


def manifest_base_path(root, experiment_id, dataset, base, checksums):
    """Select the manifest's identity, not a filesystem fallback."""
    current = base_forecast_path(root, experiment_id, dataset, base)
    historical = (Path(root) / 'results/experiments' / experiment_id
                  / 'base_forecasts' / f'{dataset}_{base}.npz')
    registered = [p for p in (historical, current)
                  if p.relative_to(root).as_posix() in checksums]
    if len(registered) != 1:
        raise ValueError('Expected exactly one base path in source manifest')
    return registered[0]
