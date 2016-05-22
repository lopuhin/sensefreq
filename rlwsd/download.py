import os.path
import tarfile

from clint.textui import progress
import requests


MODELS_URL = 'https://s3-eu-west-1.amazonaws.com/sensefreq/models.tar.gz'


def download():
    root = os.path.dirname(__file__)
    archive_filename = os.path.join(root, 'models.tar.gz')
    print('Downloading...')
    with open(archive_filename, 'wb') as f:
        r = requests.get(MODELS_URL, stream=True)
        total_length = int(r.headers.get('content-length'))
        chunk_size = 1024**2
        for chunk in progress.bar(
                r.iter_content(chunk_size=chunk_size),
                expected_size=int(total_length / chunk_size) + 1):
            if chunk:
                f.write(chunk)
                f.flush()
    print('Extracting...')
    with tarfile.open(archive_filename, 'r:gz') as f:
        f.extractall(path=root)
    print('Done!')


download()
