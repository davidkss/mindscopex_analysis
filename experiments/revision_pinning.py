"""Offline revision evidence checks; never downloads files or loads weights."""
from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.trace_fresh_v2_first_divergence import digest
from mindscopex_analysis.full_answer_protocols import token_ids_sha256

ROLES = {'model', 'tokenizer', 'sae'}
RUNTIME_FIELDS = ('runtime_tokenizer_validation', 'runtime_model_loading_validation',
                  'runtime_sae_loading_validation')
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')


def validate_revision_ids(revisions):
    if (not isinstance(revisions, dict) or set(revisions) != ROLES
            or any(not isinstance(v, str) or re.fullmatch('[0-9a-f]{40}', v) is None
                   for v in revisions.values())):
        raise ValueError('revisions require exactly model/tokenizer/sae immutable 40-character lowercase hex SHAs')


def revision_blockers(manifest):
    blockers = []
    try:
        validate_revision_ids(manifest.get('revisions'))
    except ValueError as exc:
        blockers.append(str(exc))
    state = manifest.get('revision_validation', {})
    if not isinstance(state, dict):
        state = {}
    if state.get('validated_revisions') != manifest.get('revisions'):
        blockers.append('revision validation evidence is missing/stale for these revisions')
    for key in ('revision_metadata_verified', 'revision_loader_forwarding_verified'):
        if state.get(key) is not True:
            blockers.append(f'{key} is not verified')
    for key in RUNTIME_FIELDS:
        if state.get(key) != 'verified':
            blockers.append(f'{key}={state.get(key, "unverified")}')
    # Keep the original gate and add explicit runtime checks; metadata/mock is insufficient.
    if manifest.get('pinning_validation') != 'verified':
        blockers.append('real loader pinning validation unresolved')
    return blockers


def inspect_tokenizer_cache(snapshot, files):
    """Compare existing tokenizer files to HF API Git blob or LFS payload metadata."""
    snapshot = Path(snapshot)
    records = []
    for name in TOKENIZER_FILES:
        path = snapshot / name
        metadata = next((f for f in files if f['rfilename'] == name), None)
        record = dict(filename=name, path=str(path), exists=path.is_file(), metadata_matches=False)
        if path.is_file() and metadata is not None:
            size = path.stat().st_size
            sha256 = digest(path)
            lfs = metadata.get('lfs')
            record.update(bytes=size, sha256=sha256, api_git_blob_id=metadata.get('blobId'))
            if lfs:
                # API blobId describes the LFS pointer, not the cached payload.
                record.update(hash_kind='lfs_sha256', expected_sha256=lfs.get('sha256'),
                              metadata_matches=(sha256 == lfs.get('sha256') and size == lfs.get('size') == metadata.get('size')))
            else:
                data = path.read_bytes()
                blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
                record.update(hash_kind='git_blob_sha1', git_blob_id=blob,
                              metadata_matches=(blob == metadata.get('blobId') and size == metadata.get('size')))

        records.append(record)
    return dict(snapshot_path=str(snapshot), files=records,
                all_tokenizer_files_match=all(r['metadata_matches'] for r in records),
                note='File presence/hash is not tokenizer execution or model loading validation')


def validate_tokenizer_inputs(tokenizer, case_inputs):
    """Use production default prompt encoding and whitespace-preserving candidate encoding."""
    records = []
    for cid, evidence in case_inputs.items():
        prompt_ids = list(tokenizer(evidence['prompt'])['input_ids'])
        candidate_ids = {label: list(tokenizer.encode(text, add_special_tokens=False))
                         for label, text in evidence['candidate_texts'].items()}
        expected_sha = token_ids_sha256(evidence['prompt_token_ids'])
        records.append(dict(case_id=cid, prompt_token_ids_sha256=token_ids_sha256(prompt_ids),
                            prompt_matches=(prompt_ids == evidence['prompt_token_ids'] and token_ids_sha256(prompt_ids) == expected_sha),
                            candidates_match=candidate_ids == evidence['candidate_token_ids'],
                            candidate_token_ids=candidate_ids))
    return dict(status='verified' if all(r['prompt_matches'] and r['candidates_match'] for r in records)
                and len(records) == 3 else 'mismatch', cases=records,
                scope='Tokenizer input comparison only; model/SAE loading is separate')


def validate_cached_tokenizer(snapshot, cache_evidence, case_inputs):
    """Optional existing-library, local-files-only tokenizer check. No package installation."""
    if not cache_evidence['all_tokenizer_files_match']:
        return dict(status='unverified', reason='Pinned tokenizer files missing or metadata mismatch')
    if importlib.util.find_spec('transformers') is None or importlib.util.find_spec('tokenizers') is None:
        return dict(status='unverified', reason='Transformers/Tokenizers not installed; cached files were not executed')
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
        result = validate_tokenizer_inputs(tokenizer, case_inputs)
        result['snapshot_path'] = str(snapshot)
        return result
    except Exception as exc:
        return dict(status='unverified', reason=f'Local tokenizer loading failed: {type(exc).__name__}: {exc}')


def validate_revision_evidence(manifest, evidence):
    """Bind selected IDs to successful main + immutable HF API metadata records."""
    revisions = manifest['revisions']
    validate_revision_ids(revisions)
    if evidence.get('selected_revisions') != revisions:
        raise ValueError('Revision evidence differs from manifest revisions')
    required_files = {'model': ('config.json',), 'tokenizer': TOKENIZER_FILES,
                      'sae': ('config.json', 'layer5.sae.pt')}
    for role in ROLES:
        record = evidence.get('queries', {}).get(role, {})
        repo = manifest['sae_repo_id'] if role == 'sae' else manifest['model_id']
        confirmed = record.get('immutable_confirmation', {})
        if (record.get('repo_id') != repo or record.get('sha') != revisions[role]
                or record.get('http_status') != 200 or confirmed.get('http_status') != 200
                or confirmed.get('sha') != revisions[role] or confirmed.get('file_list_matches') is not True
                or confirmed.get('url') != f'https://huggingface.co/api/models/{repo}/revision/{revisions[role]}?blobs=true'
                or not record.get('queried_at_utc') or not confirmed.get('queried_at_utc')):
            raise ValueError(f'Revision API evidence invalid: {role}')
        names = {r['rfilename'] for r in record.get('files', [])}
        if not set(required_files[role]).issubset(names):
            raise ValueError(f'Revision file metadata missing: {role}')
    if evidence['queries']['sae'].get('card_metadata', {}).get('base_model') != manifest['model_id']:
        raise ValueError('SAE Base model metadata differs')
