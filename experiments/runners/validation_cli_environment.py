"""Read-only, offline environment gate for the approved user-WSL Attempt3."""
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

CLI = Path('/home/davidkss/.local/bin/colab')
PYTHON = '/home/davidkss/.local/share/uv/tools/google-colab-cli/bin/python'
FINGERPRINTS = {
    'colab_cli/client.py': '7081d2c9decbc9a4cafb4ab89202c7e7a644f72bd74a7e846bffce5344839a06',
    'colab_cli/auth.py': '0bef52f554d75a277b91eef897ccaf0c54d5e2e6535c2b00257a6daba242e009',
    'google/auth/transport/requests.py':
        'eae46f3e87edce8ff85fb533316dcce7c4ac9cc36e27650a55cbf83b9c91bc86',
    'requests/adapters.py': '647eabd4453de232aa40b55ff033d6f19778910b7e7069c51fdf63076fc77493',
    'requests/sessions.py': '96fbb30bbbf06a59a5268d13b57885149756aa3f31695b5c15e41dd7bb2f67a6',
    'urllib3/util/retry.py': '0fdc26b67d1ecc42b0cab3972f20bb07284d783f4dc318394fe8b99098c42925',
}
VERSIONS = {'google-colab-cli': '0.7.5', 'requests': '2.34.2',
            'google-auth': '2.62.1', 'urllib3': '2.8.0'}


def verify_validation_cli_environment():
    """Never invoke CLI/auth/network. Fail before consuming a claim on drift."""
    if 'microsoft' not in platform.release().lower() or not os.environ.get('WSL_DISTRO_NAME'):
        raise ValueError('Attempt3 allocation requires the approved user WSL environment')
    if shutil.which('colab') != str(CLI):
        raise ValueError('Attempt3 CLI path differs from preflight')
    if CLI.read_text().splitlines()[0] != '#!' + PYTHON:
        raise ValueError('Attempt3 CLI interpreter differs from preflight')
    if os.environ.get('PYTHONPATH') or os.environ.get('PYTHONHOME'):
        raise ValueError('Attempt3 refuses Python import overrides')
    for key in ('REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE'):
        value = os.environ.get(key)
        if value and not Path(value).is_file():
            raise ValueError('Attempt3 CA bundle is unavailable')
    # Match normal entrypoint site resolution, but import only standard-library metadata.
    names = repr(list(VERSIONS))
    paths = repr(list(FINGERPRINTS))
    probe = '''import hashlib, importlib.metadata as m, json
from pathlib import Path
root = Path(m.distribution('google-colab-cli').locate_file(''))
versions = {name:m.version(name) for name in NAMES}
files = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in PATHS}
print(json.dumps({'versions':versions, 'files':files}))
'''.replace('NAMES', names).replace('PATHS', paths)
    result = subprocess.run([PYTHON, '-B', '-c', probe], check=True,
                            capture_output=True, text=True)
    if json.loads(result.stdout) != {'versions': VERSIONS, 'files': FINGERPRINTS}:
        raise ValueError('Attempt3 installed CLI/transport differs from preflight')
