import numpy as np
import weakref
import contextlib
import kdezero

try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray)


class Config:
    """kdezero config class

    Attributes:
        enable_backprop (bool):
            Decide if you want back propagation (default: True)
        train (bool):
            Flag of whether you are training (default: True)
    """
    enable_backprop = True
    train = True


@contextlib.contextmanager
def using_config(name, value):
    """Change Config

    Can be used with 'with' syntax to temporarily change Config

    Args:
        name (string): Attribute of Config
        value: Config value to change temporarily

    """
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    """Turn off back propagation

    Can be used with 'with' syntax to temporarily turn off back propagation

    """
    return using_config('enable_backprop', False)


def test_mode():
    """Switch to test mode

    Can be used with 'with' syntax to temporarily
    switch from training mode to test mode

    """
    return using_config('train', False)


class Variable:
    """Variable of kdezero

    A kdezero variable with numpy format data and back propagation value

    Attributes:
        data (ndarray or None):
            value of variable
        name (string or None):
            name of variable (default: None)
        grad (kdezero.Variable or None):
            Derivative value obtained by back propagation
        creator (Function or None):
            In the case of a variable created by a function, that function is stored
        generation (int):
            Represent the generation of functions and generated variables

    Args:
        data (ndarray or None):
            value of variable
        name (string or None, optional):
            name of variable (default: None)

    Raises:
        TypeError:
            Exception occurs when the data is not the corresponding ndarray class.
    """
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        """Shape of the data as ndarray
        """
        return self.data.shape

    @property
    def ndim(self):
        """Dimension of the data as ndarray
        """
        return self.data.ndim

    @property
    def size(self):
        """Size of the data as ndarray
        """
        return self.data.size

    @property
    def dtype(self):
        """Dtype of the data as ndarray
        """
        return self.data.dtype

    def __len__(self):
        """Length of the data as ndarray
        """
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        """Set the function that generates this variable and set the generation

        Args:
            func (function): Function to generate this variable
        """
        self.creator = func
        self.generation = func.generation + 1

    def unchain(self):
        """Blocking backpropagation

        Set creator to None to prevent backpropagation
        """
        self.creator = None

    def cleargrad(self):
        """Reset the differential value
        """
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        """backpropagation

        Backpropagate retroactively to the functions
        and variables used to generate this variable.

        Args:
            retain_grad (bool):
                If it is Flase, delete all,
                leaving only the derivative value at the end. (default: False)
            create_graph (bool):
                Set to True if you want to calculate the derivative value
                during backpropagation. (default: False)
        """
        if self.grad is None:
            xp = kdezero.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def unchain_backward(self):
        """Go back and prevent backpropagation

        Go back and set creator to None to prevent backpropagation

        """
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()

    def reshape(self, *shape):
        """Reshape the data (supports back propagation)

        Args:
            shape (tuple or list of int, n ints):
                Shape after conversion

        Returns:
            kdezero.Variable: Shape changed variable
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return kdezero.functions.reshape(self, shape)

    def transpose(self, *axes):
        """Perform transpose (supports back propagation)

        Args:
            axes (tuple or list of int, None or n ints):
                Specify in the same way as transpose of ndarray.
                If None is specified or not specified, perform normal transpose.

        Returns:
            kdezero.Variable: The calculation result.
        """
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return kdezero.functions.transpose(self, axes)

    @property
    def T(self):
        """normal transpose.
        """
        return kdezero.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        """Caluculate the sum of the elements of the data (supports back propagation)

        Args:
            axis (None or int or tuple of ints, optional):
                Specify like ndarray.
            keepdims (bool, optional):
                If True, the sum is calculated while preserving the dimensions.

        Returns:
            kdezero.Variable: The calculation result.
        """
        return kdezero.functions.sum(self, axis, keepdims)

    def to_cpu(self):
        """Convert data for cpu
        """
        if self.data is not None:
            self.data = kdezero.cuda.as_numpy(self.data)

    def to_gpu(self):
        """Convert data for gpu
        """
        if self.data is not None:
            self.data = kdezero.cuda.as_cupy(self.data)


def setup_variable():
    Variable.__add__ = kdezero.functions.add
    Variable.__radd__ = kdezero.functions.add
    Variable.__mul__ = kdezero.functions.mul
    Variable.__rmul__ = kdezero.functions.mul
    Variable.__neg__ = kdezero.functions.neg
    Variable.__sub__ = kdezero.functions.sub
    Variable.__rsub__ = kdezero.functions.rsub
    Variable.__truediv__ = kdezero.functions.div
    Variable.__rtruediv__ = kdezero.functions.rdiv
    Variable.__pow__ = kdezero.functions.pow
    Variable.__getitem__ = kdezero.functions.get_item

    Variable.matmul = kdezero.functions.matmul
    Variable.dot = kdezero.functions.matmul


class Parameter(Variable):
    """Parameters calculated by machine learning
    """
    pass


class Function:
    """Function of kdezero

    Backpropagating function that can be used like numpy.

    Attribute:
        generation (int):
            Represent the generation of functions and generated variables
        inputs (list of kdezero.Variable):
            List of variables used to enter the function.
        outputs (list of kdezero.Variable):
            List of function return values.
    """
    def __call__(self, *inputs):
        """Calculation and creation of calculation graph.

        If Config.enable_backprop Flase,
        the calculation graph will not be created.

        Returns:
            kdezero.Variable or list of kdezero.Variable:
                Returns the return value of the function.
                If there are two or more return values, return a list.

        """
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        """Create with inherited class
        """
        raise NotImplementedError()

    def backward(self, gys):
        """Create with inherited class
        """
        raise NotImplementedError()


def as_variable(obj):
    """Convert kdezero.Variable

    Args:
        obj (kdezero.Variable, ndarray):
            Object to convert.
            If obj is kdezero.Variable return it as it is
    
    Returns:
        kdezero.Variable: Variable converted from obj.
    """
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x, array_module=np):
    """Convert ndarray

    Args:
        x (scalar or ndarray):
            Convert scalar to ndarray, otherwise return as is.
        array_module (np or cp, optional):
            Specify a numpy or cupy module. (default: np)

    Returns:
        ndarray: Array of specified modules
    """
    if np.isscalar(x):
        return array_module.array(x)
    return x
