# cluster_cinn.py

import torch
import torch.nn as nn
from FrEIA.framework import (
    InputNode, ConditionNode, Node, OutputNode, ReversibleGraphNet
)
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

class cINN(nn.Module):
    """
    A conditional INN with GLOW‐style affine coupling blocks.
    
    We model p(y | x) by learning an invertible map y <-> z (z ~ N(0,I)), 
    conditioned on x via a small MLP (inside each coupling block).
    """
    def __init__(self,
                 y_dim:int,
                 x_dim:int,
                 hidden_dim:int=128,
                 n_blocks:int=12,
                 clamp:float=2.0):
        """
        Parameters
        ----------
        y_dim      Number of target dimensions (D_tar)
        x_dim      Number of condition dimensions (D_obs)
        hidden_dim Size of the hidden layers in each s/t subnet
        n_blocks   Number of coupling layers
        clamp      Clamp value for the GLOW coupling (controls scale stability)
        """
        super().__init__()
        # 1) Build the conditioning node (for x)
        cond_node = ConditionNode(x_dim, name='cond')

        # 2) Build the main y‐input node
        nodes = [InputNode(y_dim, name='y_in')]

        # 3) Define the small MLP used inside each coupling block
        def subnet_constructor(ch_in, ch_out):
            return nn.Sequential(
                nn.Linear(ch_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, ch_out)
            )
        def subnet_constructor(ch_in, ch_out):
            return nn.Sequential(
                nn.Linear(ch_in, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),          # ← lightweight dropout
                nn.Linear(hidden_dim, ch_out)
            )
        # 4) Stack GLOW‐style coupling + permutation layers
        for i in range(n_blocks):
            # 4a) Affine coupling conditioned on x
            nodes.append(
                Node(
                    nodes[-1],
                    GLOWCouplingBlock,
                    {'subnet_constructor': subnet_constructor,
                     'clamp': clamp},
                    conditions=cond_node,
                    name=f'coupling_{i}'
                )
            )
            # 4b) Random channel permutation for mixing
            nodes.append(
                Node(
                    nodes[-1],
                    PermuteRandom,
                    {'seed': i},
                    name=f'permute_{i}'
                )
            )

        # 5) Final output node
        nodes.append(OutputNode(nodes[-1], name='y_out'))

        # 6) Assemble into a reversible graph (include the cond_node)
        self.flow = ReversibleGraphNet(nodes + [cond_node], verbose=False)

    def forward(self, y: torch.Tensor, x: torch.Tensor):
        """
        Forward pass y → (z, log|det J|) given condition x.
        
        Returns
        -------
        z        Standard‐normal latent of shape (batch, y_dim)
        log_jac  log‐abs‐determinant of the Jacobian, shape (batch,)
        """
        return self.flow(y, c=x)

    def inverse(self, z: torch.Tensor, x: torch.Tensor):
        """
        Inverse pass z → y given condition x.
        
        Returns
        -------
        y_pred   Reconstructed y of shape (batch, y_dim)
        """
        return self.flow(z, c=x, rev=True)



#Imports:

#PyTorch (torch, torch.nn) for neural nets.
#FrEIA’s graph‐INN API:
    #InputNode, ConditionNode, Node, OutputNode, ReversibleGraphNet
    #GLOWCouplingBlock (affine coupling) and PermuteRandom (channel shuffle).

#Class cINN
#__init__:

#1. Creates a ConditionNode for your observables x.
#2. Creates an InputNode for your targets y.
#3. Defines subnet_constructor, a small 2‐layer MLP used inside each coupling block.
#4. Alternates GLOWCouplingBlock(…, conditions=cond_node) and a PermuteRandom for n_blocks times.
#5. Adds an OutputNode and wraps everything (plus the condition node) into a ReversibleGraphNet.

#forward(y,x)
#Pushes y → z under condition x, returning (z, log|det J|).

#inverse(z,x)
#Draws z → reconstruct y under x.