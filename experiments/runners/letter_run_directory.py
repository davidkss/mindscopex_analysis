"""One-use handoff for explicit-letter and open-ended CRT jobs.

An arbitrary job.log is not evidence that a directory is safe to reuse.
Only the bootstrap's current token, config hash and prepared state authorize it.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path

STATUS_FILE = 'letter_run_status.json'
TOKEN_ENV = 'MINDSCOPEX_LETTER_PREPARATION_TOKEN'



def settings(mode='binary_choice_letter'):
    if mode == 'binary_choice_letter':
        return STATUS_FILE, TOKEN_ENV, '.letter_job_claim'
    if mode == 'open_ended_crt':
        return 'open_ended_run_status.json', 'MINDSCOPEX_OPEN_ENDED_PREPARATION_TOKEN', '.open_ended_job_claim'
    raise ValueError(f'Unsupported guarded output mode: {mode}')

def _ensure_directory(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise FileExistsError(f'Refusing non-directory/symlink letter path: {path}')
    path.mkdir(parents=True, exist_ok=True)


def _new_status(config: Path, name: str, phase: str) -> dict:
    return dict(schema=1, run_name=name, config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                token=secrets.token_hex(32), phase=phase, returncode=None)


def prepare(run_dir: Path, config: Path, *, mode='binary_choice_letter') -> str:
    """Reserve an empty directory before opening a bootstrap log."""
    status_file, _, _ = settings(mode)
    _ensure_directory(run_dir)
    if any(run_dir.iterdir()):
        raise FileExistsError(f'Refusing nonempty letter experiment directory: {run_dir}')
    status = _new_status(config, run_dir.name, 'prepared')
    with (run_dir / status_file).open('x', encoding='utf-8') as stream:
        json.dump(status, stream, indent=2)
    return status['token']


def claim(run_dir: Path, config: Path, *, mode='binary_choice_letter') -> str:
    """Allow fresh/empty directories or a verified current bootstrap handoff."""
    status_file, token_env, claim_file = settings(mode)
    _ensure_directory(run_dir)
    entries = list(run_dir.iterdir())
    if not entries:
        token = prepare(run_dir, config, mode=mode)
    else:
        token = os.environ.get(token_env)
        allowed = {status_file, 'job.log'}
        if (not token or any(p.name not in allowed or p.is_symlink() or not p.is_file()
                             for p in entries)):
            raise FileExistsError(f'Refusing to reuse letter experiment directory: {run_dir}')
    try:
        status = json.loads((run_dir / status_file).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise FileExistsError(f'Invalid letter preparation status: {run_dir}') from exc
    if (not isinstance(status, dict) or status.get('schema') != 1
            or status.get('run_name') != run_dir.name or status.get('phase') != 'prepared'
            or status.get('token') != token or status.get('returncode') is not None
            or status.get('config_sha256') != hashlib.sha256(config.read_bytes()).hexdigest()):
        raise FileExistsError(f'Unverified or already used letter directory: {run_dir}')
    # Exclusive claim also prevents two processes consuming the same handoff.
    with (run_dir / claim_file).open('x', encoding='utf-8') as stream:
        stream.write(token)
    status['phase'] = 'running'
    (run_dir / status_file).write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
    return token


def finish(run_dir: Path, token: str, returncode: int, *, mode='binary_choice_letter') -> None:
    """Persist process outcome inside the directory that will be archived."""
    status_file, _, _ = settings(mode)
    path = run_dir / status_file
    if path.is_symlink():
        raise FileExistsError(f'Refusing symlink status: {path}')
    status = json.loads(path.read_text(encoding='utf-8'))
    if status.get('token') != token:
        raise ValueError('Letter execution status token mismatch')
    status.update(phase='succeeded' if returncode == 0 else 'failed', returncode=returncode)
    path.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
