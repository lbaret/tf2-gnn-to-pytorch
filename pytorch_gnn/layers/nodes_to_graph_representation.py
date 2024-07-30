"""Graph representation aggregation layer."""
from abc import abstractmethod
from typing import List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import segment_coo

from ..models.mlp import MLP
from ..utils import get_activation_function_by_name, unsorted_segment_softmax


class NodesToGraphRepresentationInput(NamedTuple):
    """A named tuple to hold input to layers computing graph representations from nodes
    representations."""

    node_embeddings: torch.Tensor
    node_to_graph_map: torch.Tensor
    num_graphs: torch.Tensor


class NodesToGraphRepresentation(nn.Module):
    """Abstract class to compute graph representations from node representations.

    Throughout we use the following abbreviations in shape descriptions:
        * V: number of nodes (across all graphs)
        * VD: node representation dimension
        * G: number of graphs
        * GD: graph representation dimension
    """

    def __init__(self, graph_representation_size: int, **kwargs):
        super().__init__(**kwargs)
        self._graph_representation_size = graph_representation_size

    @abstractmethod
    def forward(self, inputs: NodesToGraphRepresentationInput, training: bool = False):
        """Call the layer.

        Args:
            inputs: A tuple containing two items:
                node_embeddings: float32 tensor of shape [V, VD], the representation of each
                    node in all graphs.
                node_to_graph_map: int32 tensor of shape [V] with values in range [0, G-1],
                    mapping each node to a graph ID.
                num_graphs: int32 scalar, specifying the number G of graphs.
            training: A bool that denotes whether we are in training mode.

        Returns:
            float32 tensor of shape [G, GD]
        """
        pass


class WeightedSumGraphRepresentation(NodesToGraphRepresentation):
    """Layer computing graph representations as weighted sum of node representations.
    The weights are either computed from the original node representations ("self-attentional")
    or by a softmax across the nodes of a graph.
    Supports splitting operation into parallely computed independent "heads" which can focus
    on different aspects.

    Throughout we use the following abbreviations in shape descriptions:
        * V: number of nodes (across all graphs)
        * VD: node representation dimension
        * G: number of graphs
        * GD: graph representation dimension
        * H: number of heads
    """

    def __init__(
        self,
        mlp_input_size: int,
        graph_representation_size: int,
        num_heads: int,
        weighting_fun: str = "softmax",  # One of {"softmax", "sigmoid"}
        scoring_mlp_layers: List[int] = [128],
        scoring_mlp_activation_fun: str = "ReLU",
        scoring_mlp_use_biases: bool = False,
        scoring_mlp_dropout_rate: float = 0.2,
        transformation_mlp_layers: List[int] = [128],
        transformation_mlp_activation_fun: str = "ReLU",
        transformation_mlp_use_biases: bool = False,
        transformation_mlp_dropout_rate: float = 0.2,
        transformation_mlp_result_lower_bound: Optional[float] = None,
        transformation_mlp_result_upper_bound: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            mlp_input_size: Size of input features dimension.
            graph_representation_size: Size of the computed graph representation.
            num_heads: Number of independent heads to use to compute weights.
            weighting_fun: "sigmoid" ([0, 1] weights for each node computed from its
                representation), "softmax" ([0, 1] weights for each node computed
                from all nodes in same graph), "average" (weight is fixed to 1/num_nodes),
                or "none" (weight is fixed to 1).
            scoring_mlp_layers: MLP layer structure for computing raw scores turned into
                weights.
            scoring_mlp_activation_fun: MLP activcation function for computing raw scores
                turned into weights.
            scoring_mlp_dropout_rate: MLP inter-layer dropout rate for computing raw scores
                turned into weights.
            transformation_mlp_layers: MLP layer structure for computing graph representations.
            transformation_mlp_activation_fun: MLP activcation function for computing graph
                representations.
            transformation_mlp_dropout_rate: MLP inter-layer dropout rate for computing graph
                representations.
            transformation_mlp_result_lower_bound: Lower bound that results of the transformation
                MLP will be clipped to before being scaled and summed up.
                This is particularly useful to limit the magnitude of results when using "sigmoid"
                or "none" as weighting function.
            transformation_mlp_result_upper_bound: Upper bound that results of the transformation
                MLP will be clipped to before being scaled and summed up.
        """
        super().__init__(graph_representation_size, **kwargs)
        assert (
            graph_representation_size % num_heads == 0
        ), f"Number of heads {num_heads} needs to divide final representation size {graph_representation_size}!"
        assert weighting_fun.lower() in {
            "none",
            "average",
            "softmax",
            "sigmoid",
        }, f"Weighting function {weighting_fun} unknown, {{'softmax', 'sigmoid', 'none', 'average'}} supported."

        self._num_heads = num_heads
        self._weighting_fun = weighting_fun.lower()
        self._transformation_mlp_activation_fun = get_activation_function_by_name(
            transformation_mlp_activation_fun
        )
        self._transformation_mlp_result_lower_bound = transformation_mlp_result_lower_bound
        self._transformation_mlp_result_upper_bound = transformation_mlp_result_upper_bound

        # Build sub-layers:
        if self._weighting_fun not in ("none", "average"):
            self._scoring_mlp = MLP(
                in_size=mlp_input_size,
                out_size=self._num_heads,
                hidden_layers=scoring_mlp_layers,
                use_biases=scoring_mlp_use_biases,
                activation_fun=get_activation_function_by_name(
                    scoring_mlp_activation_fun
                ),
                dropout_rate=scoring_mlp_dropout_rate,
                name="ScoringMLP",
            )

        self._transformation_mlp = MLP(
            in_size=mlp_input_size,
            out_size=self._graph_representation_size,
            hidden_layers=transformation_mlp_layers,
            use_biases=transformation_mlp_use_biases,
            activation_fun=self._transformation_mlp_activation_fun,
            dropout_rate=transformation_mlp_dropout_rate,
            name="TransformationMLP",
        )
        
    def forward(self, inputs: NodesToGraphRepresentationInput) -> torch.Tensor: # NOTE: https://www.tensorflow.org/api_docs/python/tf/keras/Layer => build (in TF) + forward (PyTorch equivalent)
        # (1) compute weights for each node/head pair:
        if self._weighting_fun not in ("none", "average"):
            scores = self._scoring_mlp(inputs.node_embeddings)  # Shape [V, H]
            if self._weighting_fun == "sigmoid":
                weights = F.sigmoid(scores)  # Shape [V, H]
            elif self._weighting_fun == "softmax":
                weights_per_head = []
                for head_idx in range(self._num_heads):
                    head_scores = scores[:, head_idx]  # Shape [V]
                    head_weights = unsorted_segment_softmax(
                        logits=head_scores,
                        segment_ids=inputs.node_to_graph_map,
                        num_segments=inputs.num_graphs,
                    )  # Shape [V]
                    weights_per_head.append(head_weights.unsqueeze(-1))
                weights = torch.concat(weights_per_head, dim=1)  # Shape [V, H]
            else:
                raise ValueError()

        # (2) compute representations for each node/head pair:
        node_reprs = self._transformation_mlp_activation_fun(
            self._transformation_mlp(inputs.node_embeddings)
        )  # Shape [V, GD]
        if self._transformation_mlp_result_lower_bound is not None:
            node_reprs = torch.maximum(node_reprs, self._transformation_mlp_result_lower_bound)
        if self._transformation_mlp_result_upper_bound is not None:
            node_reprs = torch.minimum(node_reprs, self._transformation_mlp_result_upper_bound)
        node_reprs = node_reprs.reshape(
            shape=(-1, self._num_heads, self._graph_representation_size // self._num_heads),
        )  # Shape [V, H, GD//H]

        # (3) if necessary, weight representations and aggregate by graph:
        if self._weighting_fun == "none":
            node_reprs = node_reprs.reshape(
                shape=(-1, self._graph_representation_size)
            )  # Shape [V, GD]
            graph_reprs = segment_coo(
                src=node_reprs, index=inputs.node_to_graph_map, reduce="sum"
            ) # Shape [G, GD]
        elif self._weighting_fun == "average":
            node_reprs = node_reprs.reshape(
                shape=(-1, self._graph_representation_size)
            )  # Shape [V, GD]
            graph_reprs = segment_coo(
                src=node_reprs, index=inputs.node_to_graph_map, reduce="mean"
            )  # Shape [G, GD]
        else:
            weights = weights.unsqueeze(-1)  # Shape [V, H, 1]
            weighted_node_reprs = weights * node_reprs  # Shape [V, H, GD//H]

            weighted_node_reprs = weighted_node_reprs.reshape(
                shape=(-1, self._graph_representation_size)
            )  # Shape [V, GD]
            graph_reprs = segment_coo(
                src=weighted_node_reprs, index=inputs.node_to_graph_map, reduce="sum"
            ) # Shape [G, GD]

        return graph_reprs


class WASGraphRepresentation(NodesToGraphRepresentation):
    """_W_eighted _A_verage and _S_um graph representation.
    """

    def __init__(
        self,
        mlp_input_size: int,
        graph_representation_size: int = 128,
        num_heads: int = 8,
        pooling_mlp_layers: List[int] = [128, 128],
        pooling_mlp_activation_fun: str = "elu",
        pooling_mlp_use_biases: bool = True,
        pooling_mlp_dropout_rate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(graph_representation_size, **kwargs)

        self.__weighted_avg_graph_repr_layer = WeightedSumGraphRepresentation(
            mlp_input_size=mlp_input_size,
            graph_representation_size=graph_representation_size,
            num_heads=num_heads,
            weighting_fun="softmax",
            scoring_mlp_layers=pooling_mlp_layers,
            scoring_mlp_dropout_rate=pooling_mlp_dropout_rate,
            scoring_mlp_use_biases=pooling_mlp_use_biases,
            scoring_mlp_activation_fun=pooling_mlp_activation_fun,
            transformation_mlp_layers=pooling_mlp_layers,
            transformation_mlp_dropout_rate=pooling_mlp_dropout_rate,
            transformation_mlp_use_biases=pooling_mlp_use_biases,
            transformation_mlp_activation_fun=pooling_mlp_activation_fun,
        )

        self.__weighted_sum_graph_repr_layer = WeightedSumGraphRepresentation(
            mlp_input_size=mlp_input_size,
            graph_representation_size=graph_representation_size,
            num_heads=num_heads,
            weighting_fun="sigmoid",
            scoring_mlp_layers=pooling_mlp_layers,
            scoring_mlp_dropout_rate=pooling_mlp_dropout_rate,
            scoring_mlp_use_biases=pooling_mlp_use_biases,
            scoring_mlp_activation_fun=pooling_mlp_activation_fun,
            transformation_mlp_layers=pooling_mlp_layers,
            transformation_mlp_dropout_rate=pooling_mlp_dropout_rate,
            transformation_mlp_use_biases=pooling_mlp_use_biases,
            transformation_mlp_activation_fun=pooling_mlp_activation_fun,
        )

        self.__out_projection = torch.Linear(
            in_features=2*self._graph_representation_size, out_features=graph_representation_size, bias=False,
        )

    def forward(self, inputs: NodesToGraphRepresentationInput) -> torch.Tensor:
        avg_graph_repr = self.__weighted_avg_graph_repr_layer(inputs)
        sum_graph_repr = self.__weighted_sum_graph_repr_layer(inputs)

        return self.__out_projection(
            torch.concat([avg_graph_repr, sum_graph_repr], dim=-1)
        )
