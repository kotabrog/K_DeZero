import os

cache_dir = os.path.join(os.path.expanduser('~'), '.kdezero')


from kdezero.utils.calculation_graph import get_dot_graph
from kdezero.utils.calculation_graph import plot_dot_graph

from kdezero.utils.download_file import show_progress
from kdezero.utils.download_file import get_file

from kdezero.utils.numpy_utility import sum_to
from kdezero.utils.numpy_utility import reshape_sum_backward
from kdezero.utils.numpy_utility import logsumexp

from kdezero.utils.utils import get_deconv_outsize
from kdezero.utils.utils import get_conv_outsize
from kdezero.utils.utils import pair
