from kdezero.datasets import Dataset
from kdezero.utils import get_file


class ImageNet(Dataset):
    """ImageNet labels
    Convert numbers from 0 to 999 to ImageNet labels.
    """
    def __init__(self):
        NotImplemented

    @staticmethod
    def labels():
        """Return labels

        Returns:
            Dict: key is a number from 0 to 999 and value is label.
        """
        url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
        path = get_file(url)
        with open(path, 'r') as f:
            labels = eval(f.read())
        return labels
