"""Content identities for immutable workflow inputs, without decoding raw data."""
import hashlib
from pathlib import Path
from ..tools.catalog import canonical_json


def discovery_input_hash(dataset_dir):
    if not dataset_dir:raise ValueError('Discovery requires a configured dataset directory')
    root=Path(dataset_dir).resolve();cache=root/'cache'
    if cache.is_symlink() or not cache.is_dir():raise ValueError('Reviewed DEG cache directory required')
    paths=sorted(cache.glob('*.parquet'))
    if not 1<=len(paths)<=64:raise ValueError('DEG cache file count outside supported bounds')
    records=[];total=0
    for path in paths:
        if path.is_symlink() or not path.is_file():raise ValueError('Dataset input must be a regular file')
        size=path.stat().st_size;total+=size
        if size>16*1024*1024 or total>96*1024*1024:raise ValueError('Dataset exceeds snapshot size limit')
        records.append({'name':path.name,'size':size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return hashlib.sha256(canonical_json(records).encode()).hexdigest()


def provider_identity(service):
    """Bind configured execution destinations without storing credentials/URLs."""
    if service.models.injected is not None:
        value=service.models.injected.metadata
    else:
        config=service.models.prime()
        if not config.get('base_url'):return ''  # Queue-only, bound on first execution.
        value={'provider_kind':'openai_compatible','base_url':config['base_url'].rstrip('/')}
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()
