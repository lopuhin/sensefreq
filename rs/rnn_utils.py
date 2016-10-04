import contextlib

from rlwsd.utils import mystem


def repeat_iter(it, *args, **kwargs):
    while True:
        for item in it(*args, **kwargs):
            yield item


@contextlib.contextmanager
def printing_done(msg: str):
    print(msg, end=' ', flush=True)
    yield
    print('done')


def get_pos(word):
    pos_variants = set()
    for item in mystem.analyze(word):
        for variant in item.get('analysis', []):
            if 'gr' in variant:
                pos_variants.add(variant['gr'].split(',', 1)[0])
    return pos_variants
