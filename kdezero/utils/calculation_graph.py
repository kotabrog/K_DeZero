import os
import subprocess
from kdezero.utils import cache_dir


def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)


def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt


def get_dot_graph(output, verbose=True):
    """From the variables, create text that describes the structure of the model,
    written in graphviz-compatible dot language.

    Args:
        output (kdezero.Variable):
            Variables that retroactively create graphs with backpropagation.
        verbose (bool, optional):
            Whether to display in more detail. (default: True)

    Returns:
        string: dot text
    """
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n' + txt + '}'


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    """From the variables,
    use graphviz to create an image that represents the composition of the model.

    Args:
        output (kdezero.Variable):
            Variables that retroactively create graphs with backpropagation.
        verbose (bool, optional):
            Whether to display in more detail. (default: True)
        to_file (string, optional):
            File name to output. (default: 'graph.png')

    Note:
        Display the image if it can be displayed using a notebook etc.
    """
    dot_graph = get_dot_graph(output, verbose)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    graph_path = os.path.join(cache_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

    try:
        from IPython import display
        return display.Image(filename=to_file)
    except:
        pass
