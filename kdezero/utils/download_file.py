import os
import urllib.request
from kdezero.utils import cache_dir


def show_progress(block_num, block_size, total_size):
    """function to display progress bar for urllib.request.urlretrieve.
    """
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0: p = 100.0
    if i >= 30: i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')


def get_file(url, file_name=None, file_dir=cache_dir):
    """Download a file from the 'url' if it is not in the cache.

    By default, the file at the 'url' is downloaded to the '~/.kdezero'.
    Args:
        url (str): URL of the file.
        file_name (str, optional):
            Name of the file. It 'None' is specified the original
            file name is used.
        file_dir (str, optional):
            Output directory.

    Returns:
        str: path to the saved file. (cache_dir / file_name)
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(file_dir, file_name)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except(Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path
