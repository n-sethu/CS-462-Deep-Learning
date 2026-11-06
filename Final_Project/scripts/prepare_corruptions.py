"""Verify the official archive and extract only the preregistered corruptions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive', default='data/CIFAR-10-C.tar.partial')
args = parser.parse_args()
archive = Path(args.archive)
expected = '56bf5dcef84df0e2308c6dcbcbbd8499'
digest = hashlib.md5()
with archive.open('rb') as handle:
    for block in iter(lambda: handle.read(1024*1024), b''):
        digest.update(block)
if digest.hexdigest() != expected:
    raise ValueError('Official CIFAR-10-C archive checksum mismatch')
output = Path('data/CIFAR-10-C')
output.mkdir(exist_ok=True)
with tarfile.open(archive) as handle:
    for name in ('labels.npy', 'gaussian_noise.npy', 'motion_blur.npy', 'jpeg_compression.npy'):
        member = handle.getmember(f'CIFAR-10-C/{name}')
        if not member.isfile():
            raise ValueError('Expected regular dataset file')
        with handle.extractfile(member) as source, (output / name).open('wb') as dest:
            shutil.copyfileobj(source, dest)
(output / 'source.json').write_text(json.dumps(dict(url='https://zenodo.org/records/2535967/files/CIFAR-10-C.tar', md5=expected), indent=2))
archive.rename(archive.with_suffix(''))
print('Official archive verified; selected corruption arrays extracted.')
