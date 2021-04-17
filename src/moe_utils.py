# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for creating Sparsely-Gated Mixture-of-Experts Layers.

See "Outrageously Large Neural Networks"
https://arxiv.org/abs/1701.06538
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math

# Dependency imports

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow.contrib.distributions as tfd

from tensorflow.python.eager import context
from tensorflow.python.framework import function


DEFAULT_DEV_STRING = "existing_device"


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
    """Identity operation whose gradient is converted to a `Tensor`.

  Currently, the gradient to `tf.concat` is particularly expensive to
  compute if dy is an `IndexedSlices` (a lack of GPU implementation
  forces the gradient operation onto CPU).  This situation occurs when
  the output of the `tf.concat` is eventually passed to `tf.gather`.
  It is sometimes faster to convert the gradient to a `Tensor`, so as
  to get the cheaper gradient for `tf.concat`.  To do this, replace
  `tf.concat(x)` with `convert_gradient_to_tensor(tf.concat(x))`.

  Args:
    x: A `Tensor`.

  Returns:
    The input `Tensor`.
  """
    return x


def add_scope(scope=None, scope_fn=None):
    """Return a decorator which add a TF name/variable scope to a function.

  Note that the function returned by the decorator accept an additional 'name'
  parameter, which can overwritte the name scope given when the function is
  created.

  Args:
    scope (str): name of the scope. If None, the function name is used.
    scope_fn (fct): Either tf.name_scope or tf.variable_scope

  Returns:
    fct: the add_scope decorator
  """

    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            name = kwargs.pop("name",
                              None)  # Python 2 hack for keyword only args
            with scope_fn(name or scope or f.__name__):
                return f(*args, **kwargs)

        return decorated

    return decorator


def add_var_scope(scope=None):
    return add_scope(scope, scope_fn=tf.variable_scope)


def add_name_scope(scope=None):
    return add_scope(scope, scope_fn=tf.name_scope)


def _add_variable_proxy_methods(var, proxy_tensor):
    """Proxy methods of underlying variable.

  This enables our custom getters to still work with, e.g., batch norm.

  Args:
    var: Variable to proxy
    proxy_tensor: Tensor that is identity of var
  """
    proxy_tensor.read_value = lambda: tf.identity(proxy_tensor)
    proxy_tensor.assign_sub = var.assign_sub


class Parallelism(object):
    """Helper class for creating sets of parallel function calls.

  The purpose of this class is to replace this code:

      e = []
      f = []
      for i in xrange(len(devices)):
        with tf.device(devices[i]):
          e_, f_ = func(a[i], b[i], c)
          e.append(e_)
          f.append(f_)

  with this code:

      e, f = expert_utils.Parallelism(devices)(func, a, b, c)
  """

    def __init__(self,
                 device_names_or_functions,
                 reuse=True,
                 caching_devices=None,
                 daisy_chain_variables=False,
                 ps_devices=None):
        """Create a Parallelism.

    Args:
      device_names_or_functions: A list of length n, containing device names
        or device functions (see `tf.device`)
      reuse: True or None.  Whether to reuse variables created in the first
        replica in the subsequent replicas.
      caching_devices: Either `None`, or a list of length n containing device
        names.
      daisy_chain_variables: a boolean - if true, then copies variables in a
        daisy chain between devices.
      ps_devices: list<str>, list of devices for experts.

    Returns:
      a Parallelism.
    """
        assert device_names_or_functions
        self._devices = device_names_or_functions
        self._n = len(device_names_or_functions)
        self._reuse = reuse
        self._caching_devices = self._maybe_repeat(caching_devices)
        self._daisy_chain_variables = daisy_chain_variables
        self._ps_devices = ps_devices or [""]

    def __call__(self, fn, *args, **kwargs):
        """A parallel set of function calls (using the specified devices).

    Args:
      fn: a function or a list of n functions.
      *args: additional args.  Each arg should either be not a list, or a list
         of length n.
      **kwargs: additional keyword args.  Each arg should either be not a
         list, or a list of length n.

    Returns:
      either a single list of length n (if fn does not return a tuple), or a
      tuple of lists of length n (if fn returns a tuple).
    """
        # Construct lists or args and kwargs for each function.
        if args:
            my_args = transpose_list_of_lists(
                [self._maybe_repeat(arg) for arg in args])
        else:
            my_args = [[] for _ in xrange(self.n)]
        my_kwargs = [{} for _ in xrange(self.n)]
        for k, v in six.iteritems(kwargs):
            vals = self._maybe_repeat(v)
            for i in xrange(self.n):
                my_kwargs[i][k] = vals[i]

        # Construct lists of functions.
        fns = self._maybe_repeat(fn)

        # Now make the parallel call.
        outputs = []
        cache = {}
        tensor_to_var = {}
        for i in xrange(self.n):

            def daisy_chain_getter(getter, name, *args, **kwargs):
                """Get a variable and cache in a daisy chain."""
                device_var_key = (self._devices[i], name)
                if device_var_key in cache:
                    # if we have the variable on the correct device, return it.
                    return cache[device_var_key]
                if name in cache:
                    # if we have it on a different device, copy it from the last device
                    last_device_v = cache[name]
                    var = tensor_to_var[last_device_v]
                    v = tf.identity(last_device_v)
                else:
                    var = getter(name, *args, **kwargs)
                    v = tf.identity(var._ref())  # pylint: disable=protected-access

                # keep track of the original variable
                tensor_to_var[v] = var
                _add_variable_proxy_methods(tensor_to_var[v], v)
                # update the cache
                cache[name] = v
                cache[device_var_key] = v
                return v

            # Variable scope will not reset caching_device on reused variables,
            # so we make a custom getter that uses identity to cache the variable.
            # pylint: disable=cell-var-from-loop
            def caching_getter(getter, name, *args, **kwargs):
                """Cache variables on device."""
                key = (self._caching_devices[i], name)
                if key in cache:
                    return cache[key]

                v = getter(name, *args, **kwargs)
                with tf.device(self._caching_devices[i]):
                    ret = tf.identity(v._ref())  # pylint: disable=protected-access
                _add_variable_proxy_methods(v, ret)
                cache[key] = ret
                return ret

            if self._daisy_chain_variables:
                custom_getter = daisy_chain_getter
            elif self._caching_devices[i]:
                custom_getter = caching_getter
            else:
                custom_getter = None
            # pylint: enable=cell-var-from-loop
            with tf.name_scope("parallel_%d" % i):
                with tf.variable_scope(
                        tf.get_variable_scope()
                        if self._reuse else "parallel_%d" % i,
                        reuse=True if i > 0 and self._reuse else None,
                        caching_device=self._caching_devices[i],
                        custom_getter=custom_getter):
                    # TODO(noam, epot, avaswani)
                    # Allows for passing no device in case you want to default to the
                    # existing device. This is needed when we put all experts on a single
                    # device, for example in local_moe.
                    if self._devices[i] != DEFAULT_DEV_STRING:
                        with tf.device(self._devices[i]):
                            outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
                    else:
                        outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
        if isinstance(outputs[0], tuple):
            outputs = list(zip(*outputs))
            outputs = tuple([list(o) for o in outputs])
        return outputs

    @property
    def n(self):
        return self._n

    @property
    def devices(self):
        return self._devices

    @property
    def ps_devices(self):
        return self._ps_devices

    def _maybe_repeat(self, x):
        """Utility function for processing arguments that are singletons or lists.

    Args:
      x: either a list of self.n elements, or not a list.

    Returns:
      a list of self.n elements.
    """
        if isinstance(x, list):
            assert len(x) == self.n
            return x
        else:
            return [x] * self.n


def _rowwise_unsorted_segment_sum(values, indices, n):
    """UnsortedSegmentSum on each row.

  Args:
    values: a `Tensor` with shape `[batch_size, k]`.
    indices: an integer `Tensor` with shape `[batch_size, k]`.
    n: an integer.
  Returns:
    A `Tensor` with the same type as `values` and shape `[batch_size, n]`.
  """
    batch, k = tf.unstack(tf.shape(indices), num=2)
    indices_flat = tf.reshape(indices,
                              [-1]) + tf.div(tf.range(batch * k), k) * n
    ret_flat = tf.unsorted_segment_sum(
        tf.reshape(values, [-1]), indices_flat, batch * n)
    return tf.reshape(ret_flat, [batch, n])


def _normal_distribution_cdf(x, stddev):
    """Evaluates the CDF of the normal distribution.

  Normal distribution with mean 0 and standard deviation stddev,
  evaluated at x=x.

  input and output `Tensor`s have matching shapes.

  Args:
    x: a `Tensor`
    stddev: a `Tensor` with the same shape as `x`.

  Returns:
    a `Tensor` with the same shape as `x`.

  """
    return 0.5 * (1.0 + tf.erf(x / (math.sqrt(2) * stddev + 1e-20)))


def _prob_in_top_k(clean_values, noisy_values, noise_stddev, noisy_top_values,
                   k):
    """Helper function to NoisyTopKGating.

  Computes the probability that value is in top k, given different random noise.

  This gives us a way of backpropagating from a loss that balances the number
  of times each expert is in the top k experts per example.

  In the case of no noise, pass in None for noise_stddev, and the result will
  not be differentiable.

  Args:
    clean_values: a `Tensor` of shape [batch, n].
    noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
      normally distributed noise with standard deviation noise_stddev.
    noise_stddev: a `Tensor` of shape [batch, n], or None
    noisy_top_values: a `Tensor` of shape [batch, m].
       "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
    k: an integer.

  Returns:
    a `Tensor` of shape [batch, n].
  """
    batch = tf.shape(clean_values)[0]
    m = tf.shape(noisy_top_values)[1]
    top_values_flat = tf.reshape(noisy_top_values, [-1])
    # we want to compute the threshold that a particular value would have to
    # exceed in order to make the top k.  This computation differs depending
    # on whether the value is already in the top k.
    threshold_positions_if_in = tf.range(batch) * m + k
    threshold_if_in = tf.expand_dims(
        tf.gather(top_values_flat, threshold_positions_if_in), 1)
    is_in = tf.greater(noisy_values, threshold_if_in)
    if noise_stddev is None:
        return tf.to_float(is_in)
    threshold_positions_if_out = threshold_positions_if_in - 1
    threshold_if_out = tf.expand_dims(
        tf.gather(top_values_flat, threshold_positions_if_out), 1)
    # is each value currently in the top k.
    prob_if_in = _normal_distribution_cdf(clean_values - threshold_if_in,
                                          noise_stddev)
    prob_if_out = _normal_distribution_cdf(clean_values - threshold_if_out,
                                           noise_stddev)
    prob = tf.where(is_in, prob_if_in, prob_if_out)
    return prob


def cv_squared(x):
    """The squared coefficient of variation of a sample.

  Useful as a loss to encourage a positive distribution to be more uniform.
  Epsilons added for numerical stability.
  Returns 0 for an empty Tensor.

  Args:
    x: a `Tensor`.

  Returns:
    a `Scalar`.
  """
    epsilon = 1e-10
    float_size = tf.to_float(tf.size(x)) + epsilon
    mean = tf.reduce_sum(x) / float_size
    variance = tf.reduce_sum(tf.square(x - mean)) / float_size
    return variance / (tf.square(mean) + epsilon)


def _gates_to_load(gates):
    """Compute the true load per expert, given the gates.

  The load is the number of examples for which the corresponding gate is >0.

  Args:
    gates: a `Tensor` of shape [batch_size, n]
  Returns:
    a float32 `Tensor` of shape [n]
  """
    return tf.reduce_sum(tf.to_float(gates > 0), 0)


def _my_top_k(x, k):
    """GPU-compatible version of top-k that works for very small constant k.

  Calls argmax repeatedly.

  tf.nn.top_k is implemented for GPU, but the gradient, sparse_to_dense,
  seems not to be, so if we use tf.nn.top_k, then both the top_k and its
  gradient go on cpu.  Once this is not an issue, this function becomes
  obselete and should be replaced by tf.nn.top_k.

  Args:
    x: a 2d Tensor.
    k: a small integer.

  Returns:
    values: a Tensor of shape [batch_size, k]
    indices: a int32 Tensor of shape [batch_size, k]
  """
    if k > 10:
        return tf.nn.top_k(x, k)
    values = []
    indices = []
    depth = tf.shape(x)[1]
    for i in xrange(k):
        values.append(tf.reduce_max(x, 1))
        argmax = tf.argmax(x, 1)
        indices.append(argmax)
        if i + 1 < k:
            x += tf.one_hot(argmax, depth, -1e9)
    return tf.stack(values, axis=1), tf.to_int32(tf.stack(indices, axis=1))


def noisy_top_k_gating(x,
                       num_experts,
                       codes_size,
                       train,
                       deep_gating_architecture,
                       gating_convolutions,
                       init,
                       activation_fct,
                       batch_normalization,
                       dropout,
                       dropout_rate,
                       depth_gating,
                       gating_internal_size,
                       k=2,
                       initializer=tf.zeros_initializer(),
                       noisy_gating=True,
                       noise_epsilon=1e-2,
                       name=None):
    """Noisy top-k gating.

  See paper: https://arxiv.org/abs/1701.06538.

  Args:
    x: input Tensor with shape [batch_size, input_size]
    num_experts: an integer
    train: a boolean - we only add noise at training time.
    k: an integer - number of experts per example
    initializer: an initializer
    noisy_gating: a boolean
    noise_epsilon: a float
    name: an optional string

  Returns:
    gates: a Tensor with shape [batch_size, num_experts]
    load: a Tensor with shape [num_experts]
    logits: a Tensor with shape [batch_size, num_experts]
  """
    with tf.variable_scope(name, default_name="noisy_top_k_gating"):
        input_size = x.get_shape().as_list()[-1]

        if not deep_gating_architecture:
            w_gate = tf.get_variable("w_gate", [input_size, num_experts],
                                     tf.float32, initializer)
            clean_logits = tf.matmul(x, w_gate)

        else:
            if gating_convolutions:
                # x = tf.layers.batch_normalization(x)

                batch_size = tf.shape(x)[0]
                x = tf.expand_dims(x, -1)

                # gen_dense1 = tf.layers.conv1d(inputs=x,
                #                               filters=gating_internal_size,
                #                               kernel_size=codes_size,
                #                               kernel_initializer=init(),
                #                               name="g_conv_init",
                #                               activation=None)
                # gen_dense1 = tf.transpose(gen_dense1, [0, 2, 1])
                # if batch_normalization:
                #     gen_dense1 = tf.layers.batch_normalization(gen_dense1)
                # gen_dense1 = activation_fct(gen_dense1)
                #
                # for i in range(depth_gating):
                #     gen_dense1 = tf.layers.conv1d(inputs=gen_dense1,
                #                                   filters=gating_internal_size,
                #                                   kernel_size=gating_internal_size,
                #                                   kernel_initializer=init(),
                #                                   name="gen_conv_" + str(i))
                #     gen_dense1 = tf.transpose(gen_dense1, [0, 2, 1])
                #     if batch_normalization:
                #         gen_dense1 = tf.layers.batch_normalization(gen_dense1)
                #     gen_dense1 = activation_fct(gen_dense1)

                gen_dense1 = tf.layers.conv1d(inputs=x,
                                              filters=num_experts,
                                              kernel_size=codes_size,
                                              kernel_initializer=init(),
                                              name="gen_conv_ouput",
                                              use_bias=False)
                gen_dense1 = tf.transpose(gen_dense1, [0, 2, 1])

                clean_logits = tf.reshape(gen_dense1, shape=[batch_size, num_experts])


            else:
                # x = tf.layers.batch_normalization(x)

                # gen_dense1 = tf.layers.dense(inputs=x,
                #                              units=gating_internal_size,
                #                              activation=None,
                #                              name="gating_dense_init",
                #                              kernel_initializer=init())
                # if batch_normalization:
                #     gen_dense1 = tf.layers.batch_normalization(gen_dense1)
                # gen_dense1 = activation_fct(gen_dense1)
                #
                # if dropout:
                #     gen_dense1 = tf.layers.dropout(gen_dense1, rate=dropout_rate)
                #
                # for i in range(depth_gating):
                #     gen_dense1 = tf.layers.dense(inputs=gen_dense1,
                #                                  units=gating_internal_size,
                #                                  activation=None,
                #                                  name="gating_dense_" + str(i),
                #                                  kernel_initializer=init())
                #     if batch_normalization:
                #         gen_dense1 = tf.layers.batch_normalization(gen_dense1)
                #     gen_dense1 = activation_fct(gen_dense1)
                #
                #     if dropout:
                #         gen_dense1 = tf.layers.dropout(gen_dense1, rate=dropout_rate)

                clean_logits = tf.layers.dense(inputs=x,
                                         units=num_experts,
                                         activation=None,
                                         name='gating_output',
                                         kernel_initializer=init(),
                                         use_bias=False)



        if noisy_gating:
            w_noise = tf.get_variable("w_noise", [input_size, num_experts],
                                      tf.float32, initializer)

            raw_noise_stddev = tf.matmul(x, w_noise)
            noise_stddev = ((tf.nn.softplus(raw_noise_stddev) + noise_epsilon)
                            * (tf.to_float(train)))
            noisy_logits = clean_logits + (
                    tf.random_normal(tf.shape(clean_logits)) * noise_stddev)
            logits = noisy_logits
            if should_generate_summaries():
                tf.summary.histogram("noisy_logits", noisy_logits)
                tf.summary.histogram("noise_stddev", noise_stddev)
        else:
            logits = clean_logits

        top_logits, top_indices = _my_top_k(logits, min(k + 1, num_experts))
        top_k_logits = tf.slice(top_logits, [0, 0], [-1, k])
        top_k_indices = tf.slice(top_indices, [0, 0], [-1, k])
        top_k_gates = tf.nn.softmax(top_k_logits)
        # This will be a `Tensor` of shape `[batch_size, n]`, with zeros in the
        # positions corresponding to all but the top k experts per example.
        gates = _rowwise_unsorted_segment_sum(top_k_gates, top_k_indices,
                                              num_experts)
        if not deep_gating_architecture:
            if noisy_gating and k < num_experts:
                load = tf.reduce_sum(
                    _prob_in_top_k(clean_logits, noisy_logits, noise_stddev,
                                   top_logits, k), 0)
            else:
                load = _gates_to_load(gates)
        else:
            load = _gates_to_load(gates)
        if should_generate_summaries():
            tf.summary.histogram("importance", tf.reduce_sum(gates, 0))
            tf.summary.histogram("load", load)
        return gates, load, logits



class PadRemover(object):
    """Helper to remove padding from a tensor before sending to the experts.

  The padding is computed for one reference tensor containing the padding mask
  and then can be applied to any other tensor of shape [dim_origin,...].

  Ex:
      input = [
        [tok1, tok2],
        [tok3, tok4],
        [0, 0],
        [0, 0],
        [tok5, tok6],
        [0, 0],
      ]
      output = [
        [tok1, tok2],
        [tok3, tok4],
        [tok5, tok6],
      ]
  """

    def __init__(self, pad_mask):
        """Compute and store the location of the padding.

    Args:
      pad_mask (tf.Tensor): Reference padding tensor of shape
        [batch_size,length] or [dim_origin] (dim_origin=batch_size*length)
        containing non-zeros positive values to indicate padding location.
    """
        self.nonpad_ids = None
        self.dim_origin = None

        with tf.name_scope("pad_reduce/get_ids"):
            pad_mask = tf.reshape(pad_mask, [-1])  # Flatten the batch
            # nonpad_ids contains coordinates of zeros rows (as pad_mask is
            # float32, checking zero equality is done with |x| < epsilon, with
            # epsilon=1e-9 as standard, here pad_mask only contains positive values
            # so tf.abs would be redundant)
            self.nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))
            self.dim_origin = tf.shape(pad_mask)[:1]

    def remove(self, x):
        """Remove padding from the given tensor.

    Args:
      x (tf.Tensor): of shape [dim_origin,...]

    Returns:
      a tensor of shape [dim_compressed,...] with dim_compressed <= dim_origin
    """
        with tf.name_scope("pad_reduce/remove"):
            x_shape = x.get_shape().as_list()
            x = tf.gather_nd(
                x,
                indices=self.nonpad_ids,
            )
            if not context.in_eager_mode():
                # This is a hack but for some reason, gather_nd return a tensor of
                # undefined shape, so the shape is set up manually
                x.set_shape([None] + x_shape[1:])
        return x

    def restore(self, x):
        """Add padding back to the given tensor.

    Args:
      x (tf.Tensor): of shape [dim_compressed,...]

    Returns:
      a tensor of shape [dim_origin,...] with dim_compressed >= dim_origin. The
      dim is restored from the original reference tensor
    """
        with tf.name_scope("pad_reduce/restore"):
            x = tf.scatter_nd(
                indices=self.nonpad_ids,
                updates=x,
                shape=tf.concat(
                    [self.dim_origin, tf.shape(x)[1:]], axis=0),
            )
        return x


@add_name_scope("map_ids")
def map_ids(x, indices, map_fn):
    """Apply a function to each coordinate ids of a multidimentional tensor.

  This allows to process each sequence of a batch independently. This is
  similar to tf.map_fn but with tensor where the batch dim has been flatten.

  Warning: The indices ids have to be contigous and orderd in memory as the
  output vector for each of the ids are simply concatenated after being
  processed.
  Ex: if your indices are [0,2,2,1,2,0], the output will contains the processed
  rows in the following order: [0,0,1,2,2,2]

  Args:
    x (Tensor): The tensor to be dispatched of shape [length,...]
    indices (Tensor): A int32 tensor of size [length, 1] containing the batch
      coordinate of x
    map_fn (fct): Function called for every ids of the original tensor. Take
      as input a tensor of same rank than x and from shape [length_id,...] with
      length_id <= length. Isn't called if length_id == 0

  Returns:
    a tensor of same shape as x, where each elements has been processed
  """
    indices = tf.reshape(indices, [-1])

    t_i = tf.constant(0)
    # batch_coordinates start at 0
    t_batch_size = tf.reduce_max(indices) + 1

    # ta_stack_out will store the intermediate results for each individual id
    # As alternative to tf.TensorArray, scatter_update could potentially be used
    # but that would require an additional mutable tensor.
    ta_stack_out = tf.TensorArray(
        x.dtype,
        size=t_batch_size,
    )

    # Then we iterate over each sequence individually and compute the
    # transformation for each id
    while_condition = lambda t_i, *args: tf.less(t_i, t_batch_size)

    def body(t_i, ta_stack_out):
        """Loop body."""
        # Gather the ids
        current_ids = tf.to_int32(tf.where(tf.equal(indices, t_i)))
        t_row = tf.gather_nd(x, indices=current_ids)

        # TODO(epot): Should not call map_fn if t_row size is 0

        # Apply transformation to each id
        # Restore batch_dim=1 as most function expect [batch_dim, length, ...] as
        # input
        t_row = tf.expand_dims(t_row, axis=0)
        t_row = map_fn(t_row)
        t_row = tf.squeeze(t_row, axis=0)  # Squeeze for concatenation
        ta_stack_out = ta_stack_out.write(t_i, t_row)

        return [tf.add(t_i, 1), ta_stack_out]  # ++i

    # Run the loop, equivalent to:
    # stack_out = []
    # while i < batch_size:
    #   stack_out.expand(map_fn(x[indices==i]))
    _, ta_stack_out = tf.while_loop(while_condition, body, [t_i, ta_stack_out])

    # Merge all results
    return ta_stack_out.concat()


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.

  The purpose of this class is to create input minibatches for the
  experts and to combine the results of the experts to form a unified
  output tensor.

  There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".

  The class is initialized with a "gates" Tensor, which specifies which
  batch elements go to which experts, and the weights to use when combining
  the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.

  The inputs and outputs are all two-dimensional [batch, depth].
  Caller is responsible for collapsing additional dimensions prior to
  calling this class and reshaping the output to the original shape.
  See reshape_like().

  Example use:

  gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
  inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
  experts: a list of length `num_experts` containing sub-networks.

    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)

  The preceding code sets the output for a particular example b to:
  output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))

  This class takes advantage of sparsity in the gate matrix by including in the
  `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
  """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher.

    Args:
      num_experts: an integer.
      gates: a `Tensor` of shape `[batch_size, num_experts]`.

    Returns:
      a SparseDispatcher
    """
        self._gates = gates
        self._num_experts = num_experts

        where = tf.cast(tf.where(tf.transpose(gates) > 0), dtype=tf.int32)
        self._expert_index, self._batch_index = tf.unstack(
            where, num=2, axis=1)
        self._part_sizes_tensor = tf.reduce_sum(tf.to_int32(gates > 0), [0])
        self._nonzero_gates = tf.gather(
            tf.reshape(self._gates, [-1]),
            self._batch_index * num_experts + self._expert_index)

    @add_name_scope()
    def dispatch(self, inp):
        """Create one input Tensor for each expert.

    The `Tensor` for a expert `i` contains the slices of `inp` corresponding
    to the batch elements `b` where `gates[b, i] > 0`.

    Args:
      inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
    Returns:
      a list of `num_experts` `Tensor`s with shapes
        `[expert_batch_size_i, <extra_input_dims>]`.
    """
        inp = tf.gather(inp, self._batch_index)
        return tf.split(inp, self._part_sizes_tensor, 0, num=self._num_experts)

    @add_name_scope()
    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.

    The slice corresponding to a particular batch element `b` is computed
    as the sum over all experts `i` of the expert output, weighted by the
    corresponding gate values.  If `multiply_by_gates` is set to False, the
    gate values are ignored.

    Args:
      expert_out: a list of `num_experts` `Tensor`s, each with shape
        `[expert_batch_size_i, <extra_output_dims>]`.
      multiply_by_gates: a boolean

    Returns:
      a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
    """
        # see comments on convert_gradient_to_tensor
        stitched = convert_gradient_to_tensor(tf.concat(expert_out, 0))
        if multiply_by_gates:
            stitched *= tf.expand_dims(self._nonzero_gates, 1)
        combined = tf.unsorted_segment_sum(stitched, self._batch_index,
                                           tf.shape(self._gates)[0])
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.

    Returns:
      a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
          and shapes `[expert_batch_size_i]`
    """
        return tf.split(
            self._nonzero_gates,
            self._part_sizes_tensor,
            0,
            num=self._num_experts)

    def expert_to_batch_indices(self):
        """Batch indices corresponding to the examples in the per-expert `Tensor`s.

    Returns:
      a list of `num_experts` one-dimensional `Tensor`s with type `tf.int64`
          and shapes `[expert_batch_size_i]`
    """
        return tf.split(
            self._batch_index,
            self._part_sizes_tensor,
            0,
            num=self._num_experts)

    @property
    def part_sizes(self):
        return self._part_sizes_tensor


class DistributedSparseDispatcher(object):
    """A distributed version of SparseDispatcher.

  Instead of one batch of input examples, we simultaneously process
  a list of num_datashards batches of input examples.  The per-expert
  `Tensor`s contain a combination of examples from the different datashards.

  Each datashard is associated with a particular device and each expert is
  associated with a particular device.  All per-datashard and per-expert
  `Tensor`s are created on those devices.  There is no single-device bottleneck.
  """

    def __init__(self, data_parallelism, expert_parallelism, gates):
        """Create a DistributedSparseDispatcher.

    Args:
      data_parallelism: a Parallelism object.
      expert_parallelism: a Parallelism object.
      gates: a list of datashard_parallelism.n `Tensor`s of shapes
        `[batch_size[d], num_experts]`.

    Returns:
      a DistributedSparseDispatcher
    """
        self._gates = gates
        self._dp = data_parallelism
        self._ep = expert_parallelism
        assert len(gates) == self._dp.n
        self._dispatchers = self._dp(SparseDispatcher, self._ep.n, gates)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.

    Args:
      inp: a list of length num_datashards `Tensor`s with shapes
        `[batch_size[d], <extra_input_dims>]`.
    Returns:
      a list of `num_experts` `Tensor`s with shapes
        `[num_examples[i], <extra_input_dims>]`.
    """
        dispatched = self._dp(lambda a, b: a.dispatch(b), self._dispatchers,
                              inp)
        ret = self._ep(tf.concat, transpose_list_of_lists(dispatched), 0)
        if ret[0].dtype == tf.float32:
            # see comments on convert_gradient_to_tensor
            ret = self._ep(convert_gradient_to_tensor, ret)
        return ret

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, multiplied by the corresponding gates.

    Args:
      expert_out: a list of `num_experts` `Tensor`s, each with shape
        `[expert_batch_size_i, <extra_output_dims>]`.
      multiply_by_gates: a boolean.

    Returns:
      a list of num_datashards `Tensor`s with shapes
        `[batch_size[d], <extra_output_dims>]`.
    """
        expert_part_sizes = tf.unstack(
            tf.stack([d.part_sizes for d in self._dispatchers]),
            num=self._ep.n,
            axis=1)
        # list of lists of shape [num_experts][num_datashards]
        expert_output_parts = self._ep(tf.split, expert_out, expert_part_sizes)
        expert_output_parts_t = transpose_list_of_lists(expert_output_parts)

        def my_combine(dispatcher, parts):
            return dispatcher.combine(
                convert_gradient_to_tensor(tf.concat(parts, 0)),
                multiply_by_gates=multiply_by_gates)

        return self._dp(my_combine, self._dispatchers, expert_output_parts_t)

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.

    Returns:
      a list of `num_experts` one-dimensional `Tensor`s of type `tf.float32`.
    """
        return self._ep(
            tf.concat,
            transpose_list_of_lists(
                self._dp(lambda d: d.expert_to_gates(), self._dispatchers)), 0)


def transpose_list_of_lists(lol):
    """Transpose a list of equally-sized python lists.

  Args:
    lol: a list of lists
  Returns:
    a list of lists
  """
    assert lol, "cannot pass the empty list"
    return [list(x) for x in zip(*lol)]


def ffn_expert_fn(output_size,
                  input_size,
                  dropout,
                  dropout_rate,
                  batch_normalization,
                  init,
                  depth,
                  activation_fct,
                  convolution,
                  decoder_internal_size):
    """Returns a function that creates a feed-forward network.

  Use this function to create the expert_fn argument to distributed_moe.

  Args:
    input_size: an integer
    hidden_sizes: a list of integers
    output_size: an integer
    hidden_activation: a unary function.

  Returns:
    a unary function
  """

    def my_fn(x):
        # if convolution:
        #     batch_size = tf.shape(x)[0]
        #     x = tf.expand_dims(x, -1)
        #
        #     gen_dense1 = tf.layers.conv1d(inputs=x,
        #                                   filters=decoder_internal_size,
        #                                   kernel_size = input_size,
        #                                   kernel_initializer=init(),
        #                                   name="g_conv_init")
        #     gen_dense1 = tf.transpose(gen_dense1, [0, 2, 1])
        #
        # else:
        #     gen_dense1 = tf.layers.dense(inputs=x,
        #                                  units=decoder_internal_size,
        #                                  activation=None,
        #                                  name="gen_dense_init",
        #                                  kernel_initializer = init())
        # if batch_normalization:
        #     gen_dense1 = tf.layers.batch_normalization(gen_dense1)
        # gen_dense1 = activation_fct(gen_dense1)
        #
        # if dropout:
        #     gen_dense1 = tf.layers.dropout(gen_dense1, rate=dropout_rate)
        #
        #
        # for i in range(depth):
        #     if convolution:
        #         gen_dense1 = tf.layers.conv1d(inputs=gen_dense1,
        #                                       filters=decoder_internal_size,
        #                                       kernel_size=decoder_internal_size,
        #                                       kernel_initializer=init(),
        #                                       name="gen_conv_" + str(i))
        #         gen_dense1 = tf.transpose(gen_dense1, [0, 2, 1])
        #     else:
        #         gen_dense1 = tf.layers.dense(inputs=gen_dense1,
        #                                      units=decoder_internal_size,
        #                                      activation=None,
        #                                      name="gen_dense_" + str(i),
        #                                      kernel_initializer=init())
        #     if batch_normalization:
        #         gen_dense1 = tf.layers.batch_normalization(gen_dense1)
        #     gen_dense1 = activation_fct(gen_dense1)
        #
        #     if dropout:
        #         gen_dense1 = tf.layers.dropout(gen_dense1, rate=dropout_rate)
        #
        # if convolution:
        #     gen_output = tf.layers.conv1d(inputs=gen_dense1,
        #                                   filters=output_size,
        #                                   kernel_size=decoder_internal_size,
        #                                   kernel_initializer=init(),
        #                                   name="gen_conv_output")
        #     gen_output = tf.transpose(gen_output, [0, 2, 1])
        #     gen_output = tf.reshape(gen_output, shape=[batch_size, output_size])
        # else:
        #     gen_output = tf.layers.dense(inputs=gen_dense1,
        #                                  units=output_size,
        #                                  activation=None,
        #                                  name='gen_output',
        #                                  kernel_initializer = init())

        if convolution:
            batch_size = tf.shape(x)[0]
            x = tf.expand_dims(x, -1)

            gen_output = tf.layers.conv1d(inputs=x,
                                          filters=output_size,
                                          kernel_size=input_size,
                                          kernel_initializer=init(),
                                          name="gen_conv_output")
            gen_output = tf.transpose(gen_output, [0, 2, 1])
            gen_output = tf.reshape(gen_output, shape=[batch_size, output_size])

        else:
            gen_output = tf.layers.dense(inputs=x,
                                         units=output_size,
                                         activation=None,
                                         name='gen_output',
                                         kernel_initializer = init())

        return gen_output

    return my_fn


def ffn_expert_fn_decoder(encoder_internal_size, output_size):
    def my_fn(x):
        x = tf.layers.dense(x, encoder_internal_size, tf.nn.sigmoid)
        loc = tf.layers.dense(x, output_size)
        scale = tf.layers.dense(x, output_size, tf.nn.softplus)
        return sample(loc, scale), loc, scale
    return my_fn

tfd = tf.contrib.distributions


def sample(loc, scale):

    # Sample epsilon
    epsilon = tf.random_normal(tf.shape(scale), name='epsilon')

    # Sample latent variable
    std_encoder = tf.exp(0.5 * scale)
    z = loc + tf.multiply(std_encoder, epsilon)
    return(z)



def reshape_like(a, b):
    """Reshapes a to match the shape of b in all but the last dimension."""
    ret = tf.reshape(a, tf.concat([tf.shape(b)[:-1], tf.shape(a)[-1:]], 0))
    if not context.in_eager_mode():
        ret.set_shape(b.get_shape().as_list()[:-1] +
                      a.get_shape().as_list()[-1:])
    return ret


def flatten_all_but_last(a):
    """Flatten all dimensions of a except the last."""
    ret = tf.reshape(a, [-1, tf.shape(a)[-1]])
    if not context.in_eager_mode():
        ret.set_shape([None] + a.get_shape().as_list()[-1:])
    return ret


def distributed_moe(data_parallelism,
                    expert_devices,
                    xs,
                    train,
                    input_size,
                    expert_fn,
                    num_experts,
                    k=2,
                    loss_coef=1e-2,
                    name=None):
    """Call a distributed mixture of experts.

  Args:
    data_parallelism: a expert_utils.Parallelism object.
    expert_devices: a list of strings.  We round-robin the experts across these
      devices.
    xs: a list of input tensors, each with shape [... , input_size]
    train: a boolean scalar.
    input_size: an integer (input size for this layer)
    expert_fn: a unary function for each expert to run
       It should take a Tensor with shape [batch_size, input_size]
       and return a Tensor with shape [batch_size, output_size].
       e.g. ffn_expert_fn(...)
    num_experts: an integer - number of experts
    k: an integer - how many experts to use for each batch element
    loss_coef: a scalar - multiplier on load-balancing losses
    name: a string

  Returns:
    ys: a list of tensors.  Each Tensor has the same shape as the corresponding
      Tensor in xs, except for the last dimension, which is output_size.
    extra_training_loss: a scalar.  This should be added into the overall
      training loss of the model.  The backpropagation of this loss
      encourages all experts to be approximately equally used across a batch.
  """
    dp = data_parallelism
    # create a parallelism object for running the experts.
    #   We use the default of reuse=False.  Otherwise, the experts would all
    #   use the same variables.
    ep = Parallelism(
        [expert_devices[i % len(expert_devices)] for i in xrange(num_experts)],
        reuse=None)
    # Experts expect 2d input tensors, so flatten the batch dimension and all
    # spatial dimensions together.
    xs_flat = dp(tf.reshape, xs, [[-1, input_size]] * dp.n)
    with tf.variable_scope(name, default_name="moe"):
        # The gates indicate which batch elements go to which tensors.
        # load is a measure of approximately how many examples go to each expert
        gates, load, logits = dp(
            noisy_top_k_gating,
            xs_flat,
            num_experts,
            train,
            k,
            initializer=tf.zeros_initializer(),
            noisy_gating=True,
            noise_epsilon=1e-2)
        # This magic object helps us shuffle data between datashards and experts.
        dispatcher = DistributedSparseDispatcher(dp, ep, gates)
        expert_in = dispatcher.dispatch(xs_flat)
        expert_out = ep(expert_fn, expert_in)
        ys_flat = dispatcher.combine(expert_out)
        ys = dp(reshape_like, ys_flat, xs)
        # compute some load-balancing losses.
        load = tf.add_n(load)
        importance = tf.add_n(dp(tf.reduce_sum, gates, 0))
        loss = loss_coef * (cv_squared(importance) + cv_squared(load))
        return ys, loss



def tsne_repel(code_size, gate_logits, batch_size, p):

    nu = tf.constant(code_size - 1, dtype=tf.float32)

    sum_y = tf.reduce_sum(tf.square(gate_logits), reduction_indices=1)
    num = -2.0 * tf.matmul(gate_logits,
                           gate_logits,
                           transpose_b=True) + tf.reshape(sum_y, [-1, 1]) + sum_y
    num = num / nu

    p = p + 0.1 / batch_size
    p = p / tf.expand_dims(tf.reduce_sum(p, reduction_indices=1), 1)

    num = tf.pow(1.0 + num, -(nu + 1.0) / 2.0)
    attraction = tf.multiply(p, tf.log(num))
    attraction = -tf.reduce_sum(attraction)

    den = tf.reduce_sum(num, reduction_indices=1) - 1
    repellant = tf.reduce_sum(tf.log(den))

    return (repellant + attraction) / batch_size


import numpy as np
def compute_transition_probability(x, perplexity=30.0, tol=1e-4, max_iter=50, verbose=False):
    # x should be properly scaled so the distances are not either too small or too large

    if verbose:
        print('tSNE: searching for sigma ...')

    (n, d) = x.shape
    sum_x = np.sum(np.square(x), 1)

    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    p = np.zeros((n, n))

    # Parameterized by precision
    beta = np.ones((n, 1))
    entropy = np.log(perplexity) / np.log(2)

    # Binary search for sigma_i
    idx = range(n)
    for i in range(n):
        idx_i = list(idx[:i]) + list(idx[i+1:n])

        beta_min = -np.inf
        beta_max = np.inf

        # Remove d_ii
        dist_i = dist[i, idx_i]
        h_i, p_i = compute_entropy(dist_i, beta[i])
        h_diff = h_i - entropy

        iter_i = 0
        while np.abs(h_diff) > tol and iter_i < max_iter:
            if h_diff > 0:
                beta_min = beta[i].copy()
                if np.isfinite(beta_max):
                    beta[i] = (beta[i] + beta_max) / 2.0
                else:
                    beta[i] *= 2.0
            else:
                beta_max = beta[i].copy()
                if np.isfinite(beta_min):
                    beta[i] = (beta[i] + beta_min) / 2.0
                else:
                    beta[i] /= 2.0

            h_i, p_i = compute_entropy(dist_i, beta[i])
            h_diff = h_i - entropy

            iter_i += 1

        p[i, idx_i] = p_i

    if verbose:
        print('Min of sigma square: {}'.format(np.min(1 / beta)))
        print('Max of sigma square: {}'.format(np.max(1 / beta)))
        print('Mean of sigma square: {}'.format(np.mean(1 / beta)))

    return p

import sys
MAX_VAL = np.log(sys.float_info.max) / 2.0
def compute_entropy(dist=np.array([]), beta=1.0):
    p = -dist * beta
    shift = MAX_VAL - max(p)
    p = np.exp(p + shift)
    sum_p = np.sum(p)

    h = np.log(sum_p) - shift + beta * np.sum(np.multiply(dist, p)) / sum_p

    return h, p / sum_p



def local_moe(x,
              code,
              loc_code,
              scale_code,
              train,
              expert_fn,
              expert_fn_loc,
              expert_fn_scale,
              num_experts,
              num_markers,
              code_size,
              deep_gating_architecture,
              gating_convolutions,
              init,
              activation_fct,
              batch_normalization,
              dropout,
              dropout_rate,
              depth_gating,
              gating_internal_size,
              p_transition_prob,
              k=2,
              pass_x=True,
              pass_gates=False,
              additional_dispatch_params=None,
              noisy_gating=True,
              noise_eps=1e-2,
              name=None,
              loss_coef=1e-2,
              loss_coef_reconst_code=1,
              regularize_std_of_experts=False,
              regularize_importance=False,
              regularize_expert_distance=False,
              regularize_experts_dist_to_mean=False,
              regularize_laod_balancing=False,
              reconst_code_from_gates=False,
              regularize_entropy_on_gating_logits=False):
    """Call a local mixture of experts.

  Args:
    x: a tensors with shape [... , input_size]
    train: a boolean scalar.
    expert_fn: a function.
    num_experts: an integer - number of experts
    k: an integer - how many experts to use for each batch element
    loss_coef: a scalar - multiplier on load-balancing losses
    pass_x: a boolean. If true, x will also be dispatched to the experts.
    pass_gates: a boolean. If true, gates will be passed to experts. Might be
      necessary when dealing with sparse encoder-encoder decoder attention
    additional_dispatch_params: The extra tensors that need to be sent to each
      expert. Examples include batch batch coordinates (see
      common_attention.local_expert_attention)
    name: a string

  Returns:
    y: a tensor.  Has the same shape as x, except for the last dimension,
      which is output_size.
    extra_training_loss: a scalar.  This should be added into the overall
      training loss of the model.  The backpropagation of this loss
      encourages all experts to be approximately equally used across a batch.
  """

    with tf.variable_scope(name, default_name="local_moe"):
        x_flat = flatten_all_but_last(x)

        # The gates indicate which batch elements go to which tensors.
        # load is a measure of approximately how many examples go to each expert
        gates, load, logits = noisy_top_k_gating(
            x_flat,
            num_experts,
            code_size,
            train,
            deep_gating_architecture,
            gating_convolutions,
            init,
            activation_fct,
            batch_normalization,
            dropout,
            dropout_rate,
            depth_gating,
            gating_internal_size,
            k,
            initializer=tf.zeros_initializer(),
            noisy_gating=noisy_gating,
            noise_epsilon=noise_eps)

        gate_logits_softmax = tf.nn.softmax(logits)


        # This magic object helps us shuffle data between datashards and experts.
        dispatcher = SparseDispatcher(num_experts, gates)
        dispatcher_moe_loss = SparseDispatcher(num_experts, tf.ones(tf.shape(gates)))
        dispatcher_loc = SparseDispatcher(num_experts, tf.ones(tf.shape(gates)))
        dispatcher_loc_reconst = SparseDispatcher(num_experts, gates)
        dispatcher_scale = SparseDispatcher(num_experts, tf.ones(tf.shape(gates)))
        dispatcher_scale_reconst = SparseDispatcher(num_experts, gates)

        # Set up expert_fn arguments
        expert_kwargs = {}
        expert_kwargs_moe_loss = {}
        expert_kwargs_loc = {}
        expert_kwargs_loc_reconst = {}
        expert_kwargs_scale = {}
        expert_kwargs_scale_reconst = {}
        if pass_x:
            expert_kwargs["x"] = dispatcher.dispatch(x_flat)
            expert_kwargs_moe_loss["x"] = dispatcher_moe_loss.dispatch(x_flat)
            expert_kwargs_loc["x"] = dispatcher_loc.dispatch(gate_logits_softmax)
            expert_kwargs_loc_reconst["x"] = dispatcher_loc_reconst.dispatch(gate_logits_softmax)
            expert_kwargs_scale["x"] = dispatcher_scale.dispatch(gate_logits_softmax)
            expert_kwargs_scale_reconst["x"] = dispatcher_scale_reconst.dispatch(gate_logits_softmax)
        if pass_gates:
            expert_kwargs["gates"] = dispatcher.expert_to_gates()
            expert_kwargs_moe_loss["gates"] = dispatcher_moe_loss.expert_to_gates()
            expert_kwargs_loc["gates"] = dispatcher_loc.expert_to_gates()
            expert_kwargs_loc_reconst["gates"] = dispatcher_loc_reconst.expert_to_gates()
            expert_kwargs_scale["gates"] = dispatcher_scale.expert_to_gates()
            expert_kwargs_scale_reconst["gates"] = dispatcher_scale_reconst.expert_to_gates()
        for k, v in six.iteritems(additional_dispatch_params or {}):
            v = flatten_all_but_last(v)
            expert_kwargs[k] = dispatcher.dispatch(v)
            expert_kwargs_moe_loss[k] = dispatcher_moe_loss.dispatch(v)
            expert_kwargs_loc[k] = dispatcher_loc.dispatch(v)
            expert_kwargs_loc_reconst[k] = dispatcher_loc_reconst.dispatch(v)
            expert_kwargs_scale[k] = dispatcher_scale.dispatch(v)
            expert_kwargs_scale_reconst[k] = dispatcher_scale_reconst.dispatch(v)

        ep = Parallelism([DEFAULT_DEV_STRING] * num_experts, reuse=None)
        expert_outputs = ep(expert_fn, **expert_kwargs)

        # compute distances between expert outputs to maximize them
        with tf.variable_scope('moe_loss'):
            ep_moe_loss = Parallelism([DEFAULT_DEV_STRING] * num_experts, reuse=None)
            expert_outputs_moe_loss = ep_moe_loss(expert_fn, **expert_kwargs_moe_loss)

        expert_outputs_moe_loss = tf.map_fn(lambda x: tf.identity(x), elems=expert_outputs_moe_loss, dtype=tf.float32)
        expert_outputs_moe_loss = tf.transpose(expert_outputs_moe_loss, [1,0,2])
        expert_outputs_moe_loss_mean = tf.map_fn(lambda x: tf.reduce_mean(x, axis=0), expert_outputs_moe_loss, dtype=tf.float32)

        distance_matrix = pairwise_dist(expert_outputs_moe_loss_mean)

        # regularize std for expert outputs
        expert_outputs_moe_loss_std = tf.map_fn(lambda x: tf.keras.backend.std(x, axis=0), expert_outputs_moe_loss, dtype=tf.float32)


        y_flat = dispatcher.combine(expert_outputs)
        y = reshape_like(y_flat, x)


        loss = tf.zeros((1,), dtype=tf.float32)
        loss_code_reconst = tf.zeros((1,), dtype=tf.float32)
        # code_bottleneck = tf.placeholder(tf.float32, shape=[None, None])
        batch_size = tf.shape(x)[0]
        code_reconst = None
        if regularize_laod_balancing:
            loss += loss_coef * cv_squared(load)
        if regularize_std_of_experts:
            loss += tf.reduce_sum(expert_outputs_moe_loss_std)
        if regularize_importance:
            importance = tf.reduce_sum(gates, 0)
            loss += loss_coef * cv_squared(importance)
        if regularize_expert_distance:
            loss += tf.reduce_sum(distance_matrix)/2
        if regularize_experts_dist_to_mean:
            expert_outputs_moe_loss_mean_flatten = tf.reshape(expert_outputs_moe_loss_mean, [num_experts*num_markers, ])
            # expert_outputs_moe_loss_std_flatten = tf.reshape(expert_outputs_moe_loss_std, [num_experts*num_markers, ])

            # normal_dist_samples = tf.map_fn(lambda x: tf.distributions.Normal(x[0], x[1]), (expert_outputs_moe_loss_mean_flatten, expert_outputs_moe_loss_std_flatten), dtype=tf.distributions)

            expert_outputs_moe_loss_flatten = tf.reshape(tf.transpose(expert_outputs_moe_loss, [1,0,2]), [num_experts*num_markers, batch_size])

            expert_outputs_dist_to_mean = tf.map_fn(lambda x: dist_to(x[0], x[1]), (expert_outputs_moe_loss_flatten, expert_outputs_moe_loss_mean_flatten), dtype=tf.float32)

            loss += tf.reduce_sum(expert_outputs_dist_to_mean)
        if reconst_code_from_gates:
            # compute distances between expert outputs to maximize them
            with tf.variable_scope('expert_loc'):
                ep_loc = Parallelism([DEFAULT_DEV_STRING] * num_experts, reuse=tf.AUTO_REUSE)
                expert_outputs_loc = ep_loc(expert_fn_loc, **expert_kwargs_loc)

            with tf.variable_scope('expert_scale'):
                ep_scale = Parallelism([DEFAULT_DEV_STRING] * num_experts, reuse=tf.AUTO_REUSE)
                expert_outputs_scale = ep_scale(expert_fn_scale, **expert_kwargs_scale)

            loss_code_reconst = negative_log_likelihood_gaussian_mixture(code, gate_logits_softmax, expert_outputs_loc, expert_outputs_scale, num_experts)
            loss += loss_code_reconst

            # with tf.variable_scope('expert_loc', reuse=tf.AUTO_REUSE):
            #     ep_loc_reconst = Parallelism([DEFAULT_DEV_STRING] * num_experts, reuse=tf.AUTO_REUSE)
            #     expert_outputs_loc_reconst = ep_loc_reconst(expert_fn_loc, **expert_kwargs_loc_reconst)
            #
            #
            # with tf.variable_scope('expert_scale', reuse=tf.AUTO_REUSE):
            #     ep_scale_reconst = Parallelism([DEFAULT_DEV_STRING] * num_experts, reuse=tf.AUTO_REUSE)
            #     expert_outputs_scale_reconst = ep_scale_reconst(expert_fn_scale, **expert_kwargs_scale_reconst)

            # code_reconst = tf.map_fn(lambda x: sample(x[0], x[1]), (expert_outputs_loc_reconst, expert_outputs_scale_reconst), dtype=tf.float32)
            # code_reconst = tf.stack(code_reconst)


            # code_bottleneck = tf.layers.dense(tf.nn.softmax(tf.layers.batch_normalization(logits)), 2, use_bias=False)
            # code_reconst = tf.layers.dense(code_bottleneck, code_size, use_bias=False)

            # loc = tf.layers.dense(gates, 2, kernel_initializer=init(), use_bias=False)
            # scale = tf.layers.dense(gates, 2, tf.nn.softplus, kernel_initializer=init(), use_bias=False)
            # code_bottleneck = sample(loc, scale)
            # code_reconst = tf.layers.dense(code_bottleneck, code_size, use_bias=False)

            # input_code_reconst = tf.expand_dims(tf.nn.softmax(tf.layers.batch_normalization(logits)), -1)
            #
            # code_bottleneck = tf.layers.conv1d(inputs=input_code_reconst,
            #                                    filters=code_size-1,
            #                                    kernel_size = num_experts,
            #                                    kernel_initializer=init(),
            #                                    use_bias=False)
            # code_bottleneck = tf.transpose(code_bottleneck, [0, 2, 1])
            #
            # code_reconst = tf.layers.conv1d(inputs=code_bottleneck,
            #                                 filters=code_size,
            #                                 kernel_size=code_size-1,
            #                                 kernel_initializer=init(),
            #                                 use_bias=False)
            # code_reconst = tf.transpose(code_reconst, [0, 2, 1])
            # code_reconst = tf.reshape(code_reconst, shape=[batch_size, code_size])

            # code_bottleneck = tf.layers.dense(tf.nn.softmax(logits), np.min([num_experts, code_size-1]), kernel_initializer=init(), use_bias=False)
            # code_bottleneck = tf.layers.dense(gate_logits_softmax, 2, kernel_initializer=init(), use_bias=False)
            #
            # loc_code_reconst = tf.layers.dense(gate_logits_softmax, code_size, kernel_initializer=init(), use_bias=False)
            # scale_code_reconst = tf.layers.dense(gate_logits_softmax, code_size, tf.nn.softplus, kernel_initializer=init(), use_bias=False)
            # code_reconst = sample(loc_code_reconst, scale_code_reconst)

            # loss_code_reconst = loss_coef_reconst_code*tf.reduce_sum(tf.losses.absolute_difference(code, code_reconst))

            # dist_code = tfd.MultivariateNormalDiag(loc=loc_code, scale_diag=scale_code)
            # dist_code_reconst = tfd.MultivariateNormalDiag(loc=loc_code_reconst, scale_diag=scale_code_reconst)
            #
            # loss_code_reconst = loss_coef_reconst_code * tf.reduce_sum(tf.distributions.kl_divergence(dist_code, dist_code_reconst))
            #
            # loss += loss_code_reconst

            # loss_code_reconst = loss_coef_reconst_code * (tf.reduce_sum(tf.losses.absolute_difference(loc_code, loc_code_reconst)) + tf.losses.absolute_difference(scale_code, scale_code_reconst))
            # loss_code_reconst = loss_coef_reconst_code * tf.reduce_sum(tf.losses.absolute_difference(loc_code, loc_code_reconst))
            # loss_code_reconst += loss_coef_reconst_code * tf.reduce_sum(tf.losses.absolute_difference(code, code_reconst))
            #
            # loss_code_reconst =
            #
            #
            #
            # loss += loss_code_reconst

            # loss += loss_coef_reconst_code * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=code, logits=code_reconst))

        if regularize_entropy_on_gating_logits:
            softmax_gating_logits = tf.nn.softmax(logits)
            entropy_gating = tf.reduce_sum(-tf.reduce_sum(softmax_gating_logits * tf.log(softmax_gating_logits), axis=1))

            loss += entropy_gating

        # loss_code_reconst = tsne_repel(num_experts, gates, tf.cast(batch_size, dtype=tf.float32), p_transition_prob)
        # loss += loss_code_reconst


        loss *= loss_coef

        return y, loss, gates, load, expert_outputs, logits, distance_matrix, expert_outputs_moe_loss_mean, expert_outputs_moe_loss_std, code_reconst, loss_code_reconst, expert_outputs_loc, expert_outputs_scale


def negative_log_likelihood_gaussian_mixture(code, gates, experts_loc, experts_scale, num_expert):
    loss = tf.zeros((1,), dtype=tf.float32)

    pdf_eval = list()
    for i in range(num_expert):
        pdf_eval.append(tf.map_fn(lambda x: tfd.MultivariateNormalDiag(loc=x[0], scale_diag=x[1]).prob(x[2]), (experts_loc[i], experts_scale[i], code), dtype=tf.float32))

    pdf_eval = tf.stack(pdf_eval)

    gate_times_pdf = tf.multiply(pdf_eval, tf.transpose(gates))

    sum_components = tf.reduce_sum(gate_times_pdf, axis=0)
    sum_components_log = tf.log(tf.clip_by_value(sum_components, 1e-10, 1e10))
    loss = -tf.reduce_sum(sum_components_log)


    return(loss)


def dist_to(samples, target):
    dists = tf.map_fn(lambda x: tf.abs(target-x), samples, dtype=tf.float32)
    return(tf.reduce_mean(dists))


def pairwise_dist(X):
  """
  Computes pairwise distance between each pair of points
  Args:
    X - [N,D] matrix representing N D-dimensional vectors
  Returns:
    [N,N] matrix of (squared) Euclidean distances
  """
  x2 = tf.reduce_sum(X * X, 1, True)
  return x2 - 2 * tf.matmul(X, tf.transpose(X)) + tf.transpose(x2)

# def pairwise_dist(A):
#     """
#     Computes pairwise distances between each elements of A and each elements of B.
#     Args:
#       A,    [m,d] matrix
#       B,    [n,d] matrix
#     Returns:
#       D,    [m,n] matrix of pairwise distances
#     """
#     with tf.variable_scope('pairwise_dist'):
#         A = tf.squeeze(tf.stack(A))
#         B = tf.identity(A)
#
#         # squared norms of each row in A and B
#         na = tf.reduce_sum(tf.square(A), 1)
#         nb = tf.reduce_sum(tf.square(B), 1)
#
#         # na as a row and nb as a co"lumn vectors
#         na = tf.reshape(na, [-1, 1])
#         nb = tf.reshape(nb, [1, -1])
#
#         # return pairwise euclidead difference matrix
#         D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
#     return D



def local_moe_encoder(x,
              train,
              expert_fn,
              num_experts,
              k=1,
              loss_coef=1e-2,
              pass_x=True,
              pass_gates=False,
              additional_dispatch_params=None,
              noisy_gating=True,
              noise_eps=1e-2,
              name=None):
    """Call a local mixture of experts.

  Args:
    x: a tensors with shape [... , input_size]
    train: a boolean scalar.
    expert_fn: a function.
    num_experts: an integer - number of experts
    k: an integer - how many experts to use for each batch element
    loss_coef: a scalar - multiplier on load-balancing losses
    pass_x: a boolean. If true, x will also be dispatched to the experts.
    pass_gates: a boolean. If true, gates will be passed to experts. Might be
      necessary when dealing with sparse encoder-encoder decoder attention
    additional_dispatch_params: The extra tensors that need to be sent to each
      expert. Examples include batch batch coordinates (see
      common_attention.local_expert_attention)
    name: a string

  Returns:
    y: a tensor.  Has the same shape as x, except for the last dimension,
      which is output_size.
    extra_training_loss: a scalar.  This should be added into the overall
      training loss of the model.  The backpropagation of this loss
      encourages all experts to be approximately equally used across a batch.
  """

    with tf.variable_scope(name, default_name="local_moe"):
        x_flat = flatten_all_but_last(x)

        # The gates indicate which batch elements go to which tensors.
        # load is a measure of approximately how many examples go to each expert
        gates, load, logits = noisy_top_k_gating(
            x_flat,
            num_experts,
            train,
            k,
            initializer=tf.zeros_initializer(),
            noisy_gating=noisy_gating,
            noise_epsilon=noise_eps)
        # This magic object helps us shuffle data between datashards and experts.
        dispatcher = SparseDispatcher(num_experts, gates)

        # Set up expert_fn arguments
        expert_kwargs = {}
        if pass_x:
            expert_kwargs["x"] = dispatcher.dispatch(x_flat)
        if pass_gates:
            expert_kwargs["gates"] = dispatcher.expert_to_gates()
        for k, v in six.iteritems(additional_dispatch_params or {}):
            v = flatten_all_but_last(v)
            expert_kwargs[k] = dispatcher.dispatch(v)

        ep = Parallelism([DEFAULT_DEV_STRING] * num_experts, reuse=None)
        expert_outputs, loc, scale = ep(expert_fn, **expert_kwargs)

        y_flat = dispatcher.combine(expert_outputs)
        loc_flat = dispatcher.combine(loc)
        scale_flat = dispatcher.combine(scale)

        y = reshape_like(y_flat, x)

        # importance = tf.reduce_sum(gates, 0)
        # loss = loss_coef * (cv_squared(importance) + cv_squared(load))
        loss = loss_coef * cv_squared(load)
        return y, loss, gates, load, logits, loc_flat, scale_flat


class TruncatingDispatcher(object):
    """Helper for implementing a mixture of experts.

  A TruncatingDispatcher is useful when you need to deal with
  fixed-sized Tensors.  As opposed to a SparseDispatcher, which
  produces batches of different sizes for the different experts, the
  TruncatingDispatcher always produces batches of the same given size,
  and the results are returned stacked in one big tensor.

  In the case where an expert is over-capacity, the last items that
  should have gone to that expert are dropped.

  Confusingly, the inputs to a TruncatingDispatcher have both a
  "batch" and a "length" dimension.  Not only does each expert receive
  the same total number of examples, it also receives the same number
  of examples for each element of "batch".  This behavior is necessary
  for applications such as grouped attention, where we have a batch of
  sequences, and we want each sequence to be divided evenly among
  experts.  For simpler applications like mixture-of-experts, you can
  reshape the input so that the "batch" dimension is 1, and only the
  "length" dimension is used.
  """

    @add_name_scope("truncating_dispatcher")
    def __init__(self, requests, expert_capacity):
        """Create a TruncatingDispatcher.

    Args:
      requests: a boolean `Tensor` of shape `[batch, length, num_experts]`.
        Alternatively, a float or int Tensor containing zeros and ones.
      expert_capacity: a Scalar - maximum number of examples per expert per
        batch element.

    Returns:
      a TruncatingDispatcher
    """
        self._requests = tf.to_float(requests)
        self._expert_capacity = expert_capacity
        expert_capacity_f = tf.to_float(expert_capacity)
        self._batch, self._length, self._num_experts = tf.unstack(
            tf.shape(self._requests), num=3)

        # [batch, length, num_experts]
        position_in_expert = tf.cumsum(self._requests, axis=1, exclusive=True)
        # [batch, length, num_experts]
        self._gates = self._requests * tf.to_float(
            tf.less(position_in_expert, expert_capacity_f))
        batch_index = tf.reshape(
            tf.to_float(tf.range(self._batch)), [self._batch, 1, 1])
        length_index = tf.reshape(
            tf.to_float(tf.range(self._length)), [1, self._length, 1])
        expert_index = tf.reshape(
            tf.to_float(tf.range(self._num_experts)),
            [1, 1, self._num_experts])
        # position in a Tensor with shape [batch * num_experts * expert_capacity]
        flat_position = (position_in_expert + batch_index *
                         (tf.to_float(self._num_experts) * expert_capacity_f) +
                         expert_index * expert_capacity_f)
        # Tensor of shape [batch * num_experts * expert_capacity].
        # each element is an integer in [0, length)
        self._indices = tf.unsorted_segment_sum(
            data=tf.reshape((length_index + 1.0) * self._gates, [-1]),
            segment_ids=tf.to_int32(tf.reshape(flat_position, [-1])),
            num_segments=self._batch * self._num_experts * expert_capacity)
        self._indices = tf.reshape(
            self._indices, [self._batch, self._num_experts, expert_capacity])
        # Tensors of shape [batch, num_experts, expert_capacity].
        # each element is 0.0 or 1.0
        self._nonpadding = tf.minimum(self._indices, 1.0)
        # each element is an integer in [0, length)
        self._indices = tf.nn.relu(self._indices - 1.0)
        # self._flat_indices is [batch, num_experts, expert_capacity], with values
        # in [0, batch * length)
        self._flat_indices = tf.to_int32(self._indices + (
            tf.reshape(tf.to_float(tf.range(self._batch)), [-1, 1, 1]) *
            tf.to_float(self._length)))
        self._indices = tf.to_int32(self._indices)

    @add_name_scope("truncating_dispatcher_dispatch")
    def dispatch(self, inp):
        """Send the inputs to the experts.

    Args:
      inp: a `Tensor` of shape "[batch, length, depth]`
    Returns:
      a tensor with shape [batch, num_experts, expert_capacity, depth]
    """
        inp = tf.reshape(inp, [self._batch * self._length, -1])
        # [batch, num_experts, expert_capacity, depth]
        ret = tf.gather(inp, self._flat_indices)
        return ret

    @add_name_scope("truncating_dispatcher_combine")
    def combine(self, x):
        """Return the output from the experts.

    When one example goes to multiple experts, the outputs are summed.

    Args:
      x: a Tensor with shape [batch, num_experts, expert_capacity, depth]

    Returns:
      a `Tensor` with shape `[batch, length, depth]
    """
        depth = tf.shape(x)[-1]
        x *= tf.expand_dims(self._nonpadding, -1)
        ret = tf.unsorted_segment_sum(
            x, self._flat_indices, num_segments=self._batch * self._length)
        ret = tf.reshape(ret, [self._batch, self._length, depth])
        return ret

    def nonpadding(self):
        """Which elements of a dispatched Tensor are not padding.

    Returns:
      a Zero/One float tensor with shape [batch, num_experts, expert_capacity].
    """
        return self._nonpadding

    def gates(self):
        """A Tensor indicating which examples go to which experts.

    Returns:
      A float32 Tensor with shape [batch, length, num_experts], where each value
      is 0.0 or 1.0.
    """
        return self._gates

    def length_coordinate(self):
        """Length coordinate of dispatched tensor.

    Returns:
      a tensor with shape [batch, num_experts, expert_capacity] containing
       integers in the range [0, length)
    """
        return self._indices


def should_generate_summaries():
    """Is this an appropriate context to generate summaries.

  Returns:
    a boolean
  """
    if "while/" in tf.contrib.framework.get_name_scope():
        # Summaries don't work well within tf.while_loop()
        return False
    if tf.get_variable_scope().reuse:
        # Avoid generating separate summaries for different data shards
        return False
    return True
