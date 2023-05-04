import os
from tqdm import tqdm
from urllib.request import urlretrieve


from uphill import loggerx

from .io import touch_dir, md5sum


def tqdm_urlretrieve_hook(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> from urllib.request import urlretrieve
    >>> with tqdm(...) as t:
    ...     reporthook = tqdm_urlretrieve_hook(t)
    ...     urlretrieve(..., reporthook=reporthook)
    Source: https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] or -1,
            remains unchanged.
        """
        if tsize not in (None, -1):
            t.total = tsize
        displayed = t.update((b - last_b[0]) * bsize)
        last_b[0] = b
        return displayed

    return update_to


def urlretrieve_progress(url, filename=None, data=None, desc=None):
    """
    Works exactly like urllib.request.urlretrieve, but attaches a tqdm hook to display
    a progress bar of the download.
    Use "desc" argument to display a user-readable string that informs what is being downloaded.
    """
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
        reporthook = tqdm_urlretrieve_hook(t)
        return urlretrieve(url=url, filename=filename, reporthook=reporthook, data=data)


def download_url(url: str, md5: str, target_dir: str, save_name: str = None, force_download: bool = False):
    """Download file from url to target_dir, and check md5."""

    # target_dir = os.path.dirname(save_name)
    touch_dir(target_dir)
    filepath = os.path.join(
        target_dir,
        url.split('/')[-1] if save_name is None else save_name
    )

    md5_checked = None
    if os.path.exists(filepath):
        md5_checked = md5sum(filepath)

    if md5_checked == md5 and not force_download:
        loggerx.info('File md5 matched, skip downloading existed %s' % filepath)
        return filepath
    else:
        if os.path.exists(filepath):
            loggerx.warning('file md5 %s vs %s, not matched, re-download' % (md5_checked, md5))

        loggerx.info('Begin to download from {}'.format(url))
        
        urlretrieve_progress(url, filename=filepath, desc=f"Downloading ...")

        loggerx.info('Download finished, save file to {}'.format(filepath))
    return filepath


### TODO: how to download m3u8 url?
