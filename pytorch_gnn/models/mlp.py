"""MLP layer."""
import sys
from typing import Callable, List, Optional, Union

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_layers: Union[List[int], int] = 1,
        use_biases: bool = False,
        activation_fun: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dropout_rate: float = 0.0,
        name: str = "MLP",
    ):
        """
        Create new MLP with given number of hidden layers.

        Arguments:
            out_size: Dimensionality of output.
            hidden_layers: Either an integer determining number of hidden layers, which will have
                out_size units each; or list of integers whose lengths determines the number of
                hidden layers and whose contents the number of units in each layer.
            use_biases: Flag indicating use of bias in fully connected layers.
            activation_fun: Activation function applied between hidden layers (NB: the output of the
                MLP is always the direct result of a linear transformation)
            dropout_rate: Dropout applied to inputs of each MLP layer.
            name: Name of the MLP, used in names of created variables.
        """
        super().__init__()
        if isinstance(hidden_layers, int):
            if out_size == 1:
                print(
                    f"W: In {name}, was asked to use {hidden_layers} layers of size 1, which is most likely wrong."
                    f" Switching to {hidden_layers} layers of size 32; to get hidden layers of size 1,"
                    f" use hidden_layers=[1,...,1] explicitly.",
                    file=sys.stderr,
                )
                self._hidden_layer_sizes = [32] * hidden_layers
            else:
                self._hidden_layer_sizes = [out_size] * hidden_layers
        else:
            self._hidden_layer_sizes = hidden_layers

        if len(self._hidden_layer_sizes) > 1:
            assert (
                activation_fun is not None
            ), "Multiple linear layers without an activation"

        self._in_size = in_size
        self._out_size = out_size
        self._use_biases = use_biases
        self._activation_fun = activation_fun
        self._dropout_rate = dropout_rate
        self._layers = nn.ModuleList()
        self._name = name
        
        last_shape_dim = None
        for hidden_layer_idx, hidden_layer_size in enumerate(self._hidden_layer_sizes):
            with tf.name_scope(f"{self._name}_dense_layer_{hidden_layer_idx}"):
                self._layers.append(
                    nn.Dropout(p=self._dropout_rate),
                )
                self._layers.append(
                    nn.Linear(
                        in_features=self._in_size if last_shape_dim is None else last_shape_dim,
                        out_features=hidden_layer_size,
                        bias=self._use_biases,
                    )
                )
                if self._activation_fun is not None:
                    self._layers.append(
                        self._activation_fun,
                    )
                last_shape_dim = hidden_layer_size
        
        self._layers.append(
            nn.Linear(
                in_features=last_shape_dim,
                out_features=self._out_size,
                bias=self._use_biases,
            )
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = input
        for layer in self._layers:
            y = layer(y)
        
        return y
