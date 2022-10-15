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
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, path=root)
    print('Done!')


download()
