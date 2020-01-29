""" CNN cell for architecture search """
import torch
import torch.nn as nn
from models import ops
from models.ops import OPS


class Cell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """

    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction, genotype):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
            genotype: learned architecture for normal/reduction cells by the archtecture search step
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes
        self.genotype = genotype

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)

        self.gp = genotype[2] if reduction else genotype[0]

        self.stride = 2 if reduction else 1
        self.C = C

        # generate dag
        self.dag = nn.ModuleList()
        # for i in range(self.n_nodes):

    def forward(self, s0, s1):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)
        states = [s0, s1]

        for gp_pair in self.gp:
            op_name_1, index_1 = gp_pair[0]
            op_name_2, index_2 = gp_pair[1]
            out_1 = OPS[op_name_1](self.C,
                                   self.stride, affine=False)(states[index_1])
            out_2 = OPS[op_name_2](self.C,
                                   self.stride, affine=False)(states[index_2])
            # out = torch.cat([out_1, out_2])
            out = sum([out_1, out_2])
            states.append(out)
        s_out = torch.cat(states[2:], dim=1)
        return s_out
