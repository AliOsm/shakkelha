# Original from: https://github.com/titu1994/keras-normalized-optimizers

from __future__ import division

from keras import optimizers
from keras.legacy import interfaces
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K


def max_norm(grad):
    """
    Computes the L-infinity norm of the gradient.
    # Arguments:
        grad: gradient for a variable
    # Returns:
        The norm of the gradient
    """
    grad_max = K.max(K.abs(grad))
    norm = grad_max + K.epsilon()
    return norm


def min_max_norm(grad):
    """
    Computes the average of the Max and Min of the absolute
    values of the gradients.
    # Arguments:
        grad: gradient for a variable
    # Returns:
        The norm of the gradient
    """
    grad_min = K.min(K.abs(grad))
    grad_max = K.max(K.abs(grad))
    norm = ((grad_max + grad_min) / 2.0) + K.epsilon()
    return norm


def std_norm(grad):
    """
    Computes the standard deviation of the gradient.
    # Arguments:
        grad: gradient for a variable
    # Returns:
        The norm of the gradient
    """
    norm = K.std(grad) + K.epsilon()
    return norm


def l1_norm(grad):
    """
    Computes the L-1 norm of the gradient.
    # Arguments:
        grad: gradient for a variable
    # Returns:
        The norm of the gradient
    """
    norm = K.sum(K.abs(grad)) + K.epsilon()
    return norm


def l2_norm(grad):
    """
    Computes the L-2 norm of the gradient.
    # Arguments:
        grad: gradient for a variable
    # Returns:
        The norm of the gradient
    """
    norm = K.sqrt(K.sum(K.square(grad))) + K.epsilon()
    return norm


def l1_l2_norm(grad):
    """
    Computes the average of the L-1 and L-2 norms of the gradient.
    # Arguments:
        grad: gradient for a variable
    # Returns:
        The norm of the gradient
    """
    l1 = l1_norm(grad)
    l2 = l2_norm(grad)
    norm = ((l1 + l2) / 2.) + K.epsilon()
    return norm


def average_l1_norm(grad):
    """
    Computes the average of the L-1 norm (instead of sum) of the
    gradient.
    # Arguments:
        grad: gradient for a variable
    # Returns:
        The norm of the gradient
    """
    norm = K.mean(K.abs(grad)) + K.epsilon()
    return norm


def average_l2_norm(grad):
    """
    Computes the average of the L-2 norm (instead of sum) of the
    gradient.
    # Arguments:
        grad: gradient for a variable
    # Returns:
        The norm of the gradient
    """
    norm = K.sqrt(K.mean(K.square(grad))) + K.epsilon()
    return norm


def average_l1_l2_norm(grad):
    """
    Computes the average of the L-1 and L-2 norms (instead of the sum)
    to compute the normalized gradient.
    # Arguments:
        grad: gradient for a variable
    # Returns:
        The norm of the gradient
    """
    l1_norm = K.mean(K.abs(grad))
    l2_norm = K.sqrt(K.mean(K.square(grad)))
    norm = ((l1_norm + l2_norm) / 2.) + K.epsilon()
    return norm


class OptimizerWrapper(optimizers.Optimizer):

    def __init__(self, optimizer):
        """
        Base wrapper class for a Keras optimizer such that its gradients are
        corrected prior to computing the update ops.
        Since it is a wrapper optimizer, it must delegate all normal optimizer
        calls to the optimizer that it wraps.
        Note:
            This wrapper optimizer monkey-patches the optimizer it wraps such that
            the call to `get_gradients` will call the gradients of the
            optimizer and then normalize the list of gradients.
            This is required because Keras calls the optimizer's `get_gradients`
            method inside `get_updates`, and without this patch, we cannot
            normalize the gradients before computing the rest of the
            `get_updates` code.
        # Abstract Methods
            get_gradients: Must be overridden to support differnt gradient
                operations.
            get_config: Config needs to be carefully built for serialization.
            from_config: Config must be carefully used to build a Subclass.
        # Arguments:
            optimizer: Keras Optimizer or a string. All optimizers other
                than TFOptimizer are supported. If string, instantiates a
                default optimizer with that alias.
        # Raises
            NotImplementedError: If `optimizer` is of type `TFOptimizer`.
        """
        if optimizer.__class__.__name__ == 'TFOptimizer':
            raise NotImplementedError('Currently, TFOptimizer is not supported.')

        self.optimizer = optimizers.get(optimizer)

        # patch the `get_gradients` call
        self._optimizer_get_gradients = self.optimizer.get_gradients

    def get_gradients(self, loss, params):
        """
        Compute the gradients of the wrapped Optimizer.
        # Arguments:
            loss: Keras tensor with a single value.
            params: List of tensors to optimize
        # Returns:
            A list of normalized gradient tensors
        """
        grads = self._optimizer_get_gradients(loss, params)
        return grads

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        """
        Computes the update operations of the wrapped Optimizer using
        normalized gradients and returns a list of operations.
        # Arguments:
            loss: Keras tensor with a single value
            params: List of tensors to optimize
        # Returns:
            A list of parameter and optimizer update operations
        """
        # monkey patch `get_gradients`
        self.optimizer.get_gradients = self.get_gradients

        # get the updates
        self.optimizer.get_updates(loss, params)

        # undo monkey patch
        self.optimizer.get_gradients = self._optimizer_get_gradients

        return self.updates

    def set_weights(self, weights):
        """
        Set the weights of the wrapped optimizer by delegation
        # Arguments:
            weights: List of weight matrices
        """
        self.optimizer.set_weights(weights)

    def get_weights(self):
        """
        Get the weights of the wrapped optimizer by delegation
        # Returns:
            List of weight matrices
        """
        return self.optimizer.get_weights()

    def get_config(self):
        """
        Updates the config of the wrapped optimizer with some meta
        data about the normalization function as well as the optimizer
        name so that model saving and loading can take place
        # Returns:
            dictionary of the config
        """
        # properties of NormalizedOptimizer
        config = {'optimizer_name': self.optimizer.__class__.__name__.lower()}

        # optimizer config
        optimizer_config = {'optimizer_config': self.optimizer.get_config()}
        return dict(list(optimizer_config.items()) + list(config.items()))

    @property
    def weights(self):
        return self.optimizer.weights

    @property
    def updates(self):
        return self.optimizer.updates

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError

    @classmethod
    def set_normalization_function(cls, name, func):
        """
        Allows the addition of new normalization functions adaptively
        # Arguments:
            name: string name of the normalization function
            func: callable function which takes in a single tensor and
                returns a single tensor (input gradient tensor and output
                normalized gradient tensor).
        """
        global _NORMS
        _NORMS[name] = func

    @classmethod
    def get_normalization_functions(cls):
        """
        Get the list of all registered normalization functions that can be
        used.
        # Returns:
            list of strings denoting the names of all of the normalization
            functions.
        """
        global _NORMS
        return sorted(list(_NORMS.keys()))


class NormalizedOptimizer(OptimizerWrapper):

    def __init__(self, optimizer, normalization='l2'):
        """
        Creates a wrapper for a Keras optimizer such that its gradients are
        normalized prior to computing the update ops.
        Since it is a wrapper optimizer, it must delegate all normal optimizer
        calls to the optimizer that it wraps.
        Note:
            This wrapper optimizer monkey-patches the optimizer it wraps such that
            the call to `get_gradients` will call the gradients of the
            optimizer and then normalize the list of gradients.
            This is required because Keras calls the optimizer's `get_gradients`
            method inside `get_updates`, and without this patch, we cannot
            normalize the gradients before computing the rest of the
            `get_updates` code.
        # Arguments:
            optimizer: Keras Optimizer or a string. All optimizers other
                than TFOptimizer are supported. If string, instantiates a
                default optimizer with that alias.
            normalization: string. Must refer to a normalization function
                that is available in this modules list of normalization
                functions. To get all possible normalization functions,
                use `NormalizedOptimizer.get_normalization_functions()`.
        # Raises
            ValueError: If an incorrect name is supplied for `normalization`,
                such that the normalization function is not available or not
                set using `NormalizedOptimizer.set_normalization_functions()`.
            NotImplementedError: If `optimizer` is of type `TFOptimizer`.
        """
        super(NormalizedOptimizer, self).__init__(optimizer)

        if normalization not in _NORMS:
            raise ValueError('`normalization` must be one of %s.\n' 
                             'Provided was "%s".' % (str(sorted(list(_NORMS.keys()))), normalization))

        self.normalization = normalization
        self.normalization_fn = _NORMS[normalization]

    def get_gradients(self, loss, params):
        """
        Compute the gradients of the wrapped Optimizer, then normalize
        them with the supplied normalization function.
        # Arguments:
            loss: Keras tensor with a single value.
            params: List of tensors to optimize
        # Returns:
            A list of normalized gradient tensors
        """
        grads = super(NormalizedOptimizer, self).get_gradients(loss, params)
        grads = [grad / self.normalization_fn(grad) for grad in grads]
        return grads

    def get_config(self):
        """
        Updates the config of the wrapped optimizer with some meta
        data about the normalization function as well as the optimizer
        name so that model saving and loading can take place
        # Returns:
            dictionary of the config
        """
        # properties of NormalizedOptimizer
        config = {'normalization': self.normalization}

        # optimizer config
        base_config = super(NormalizedOptimizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        """
        Utilizes the meta data from the config to create a new instance
        of the optimizer which was wrapped previously, and creates a
        new instance of this wrapper class.
        # Arguments:
            config: dictionary of the config
        # Returns:
            a new instance of NormalizedOptimizer
        """
        optimizer_config = {'class_name': config['optimizer_name'],
                            'config': config['optimizer_config']}

        optimizer = optimizers.get(optimizer_config)
        normalization = config['normalization']

        return cls(optimizer, normalization=normalization)


class ClippedOptimizer(OptimizerWrapper):

    def __init__(self, optimizer, normalization='l2', clipnorm=1.0):
        """
        Creates a wrapper for a Keras optimizer such that its gradients are
        clipped by the norm prior to computing the update ops.
        Since it is a wrapper optimizer, it must delegate all normal optimizer
        calls to the optimizer that it wraps.
        Note:
            This wrapper optimizer monkey-patches the optimizer it wraps such that
            the call to `get_gradients` will call the gradients of the
            optimizer and then normalize the list of gradients.
            This is required because Keras calls the optimizer's `get_gradients`
            method inside `get_updates`, and without this patch, we cannot
            normalize the gradients before computing the rest of the
            `get_updates` code.
        # Arguments:
            optimizer: Keras Optimizer or a string. All optimizers other
                than TFOptimizer are supported. If string, instantiates a
                default optimizer with that alias.
            normalization: string. Must refer to a normalization function
                that is available in this modules list of normalization
                functions. To get all possible normalization functions,
                use `NormalizedOptimizer.get_normalization_functions()`.
            clipnorm: float >= 0. Gradients will be clipped
                when their norm exceeds this value.
        # Raises
            ValueError: If an incorrect name is supplied for `normalization`,
                such that the normalization function is not available or not
                set using `ClippedOptimizer.set_normalization_functions()`.
            NotImplementedError: If `optimizer` is of type `TFOptimizer`.
        """
        super(ClippedOptimizer, self).__init__(optimizer)

        if normalization not in _NORMS:
            raise ValueError('`normalization` must be one of %s.\n' 
                             'Provided was "%s".' % (str(sorted(list(_NORMS.keys()))), normalization))

        self.normalization = normalization
        self.normalization_fn = _NORMS[normalization]

        self.clipnorm = clipnorm

    def get_gradients(self, loss, params):
        """
        Compute the gradients of the wrapped Optimizer, then normalize
        them with the supplied normalization function.
        # Arguments:
            loss: Keras tensor with a single value.
            params: List of tensors to optimize
        # Returns:
            A list of normalized gradient tensors
        """
        grads = super(ClippedOptimizer, self).get_gradients(loss, params)
        grads = [self._clip_grad(grad) for grad in grads]
        return grads

    def get_config(self):
        """
        Updates the config of the wrapped optimizer with some meta
        data about the normalization function as well as the optimizer
        name so that model saving and loading can take place
        # Returns:
            dictionary of the config
        """
        # properties of NormalizedOptimizer
        config = {'normalization': self.normalization,
                  'clipnorm': self.clipnorm}

        # optimizer config
        base_config = super(ClippedOptimizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _clip_grad(self, grad):
        """
        Helper method to compute the norm and then clip the gradients.
        # Arguments:
            grad: gradients of a single variable
        # Returns:
            clipped gradients
        """
        norm = self.normalization_fn(grad)
        grad = optimizers.clip_norm(grad, self.clipnorm, norm)
        return grad

    @classmethod
    def from_config(cls, config):
        """
        Utilizes the meta data from the config to create a new instance
        of the optimizer which was wrapped previously, and creates a
        new instance of this wrapper class.
        # Arguments:
            config: dictionary of the config
        # Returns:
            a new instance of NormalizedOptimizer
        """
        optimizer_config = {'class_name': config['optimizer_name'],
                            'config': config['optimizer_config']}

        optimizer = optimizers.get(optimizer_config)
        normalization = config['normalization']
        clipnorm = config['clipnorm']

        return cls(optimizer, normalization=normalization, clipnorm=clipnorm)


_NORMS = {
    'max': max_norm,
    'min_max': min_max_norm,
    'l1': l1_norm,
    'l2': l2_norm,
    'linf': max_norm,
    'l1_l2': l1_l2_norm,
    'std': std_norm,
    'avg_l1': average_l1_norm,
    'avg_l2': average_l2_norm,
    'avg_l1_l2': average_l1_l2_norm,
}

# register this optimizer to the global custom objects when it is imported
get_custom_objects().update({'NormalizedOptimizer': NormalizedOptimizer, 'ClippedOptimizer': ClippedOptimizer})