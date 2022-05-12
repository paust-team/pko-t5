import fire

from .server import serve


fire.Fire({
    'serve': serve,
})