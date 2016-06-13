import contextlib


def repeat_iter(it, *args, **kwargs):
    while True:
        for item in it(*args, **kwargs):
            yield item


@contextlib.contextmanager
def printing_done(msg: str):
    print(msg, end=' ', flush=True)
    yield
    print('done')
