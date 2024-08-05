from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpaceGroupDenseLayer(nn.Module):
    def __init__(self, input_dim, features, num_space_groups=230):
        super(SpaceGroupDenseLayer, self).__init__()
        self.num_space_groups = num_space_groups
        self.features = features
        self.input_dim = input_dim

        # Create complex weights and biases for each space group
        self.weights_real = nn.Parameter(torch.Tensor(num_space_groups, input_dim, features))
        self.weights_imag = nn.Parameter(torch.Tensor(num_space_groups, input_dim, features))
        self.bias_real = nn.Parameter(torch.Tensor(num_space_groups, features))
        self.bias_imag = nn.Parameter(torch.Tensor(num_space_groups, features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights_real)
        nn.init.xavier_uniform_(self.weights_imag)
        nn.init.zeros_(self.bias_real)
        nn.init.zeros_(self.bias_imag)

    def forward(self, inputs, space_group):
        # Combine real and imaginary parts into complex tensors
        weights = torch.complex(self.weights_real, self.weights_imag)
        biases = torch.complex(self.bias_real, self.bias_imag)

        # Select weights and biases for the given space groups
        selected_weights = weights[space_group].squeeze(1)
        selected_biases = biases[space_group]

        outputs = torch.einsum('bni,bif->bnf', inputs, selected_weights)
        outputs = outputs + selected_biases

        return outputs

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, abc_combinations, graphs_array):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer('abc_combinations', torch.from_numpy(abc_combinations))
        self.register_buffer('graphs_array', torch.from_numpy(graphs_array))

        self.norm = nn.LayerNorm(embedding_dim)
        self.dense1 = SpaceGroupDenseLayer(input_dim=embedding_dim//2,
                                           features=embedding_dim//2)
        self.dense2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, atom_positions, reciprocal_matrices, space_group):
        # Compute reciprocal lattice points
        transformed_abc = torch.matmul(self.abc_combinations.float(), reciprocal_matrices.transpose(1, 2).float())

        # Compute encodings
        encodings = torch.exp(1j * 2 * torch.pi * torch.matmul(atom_positions, transformed_abc.transpose(1, 2)))

        # Apply adjacency matrices
        adjacency_matrices = self.graphs_array[space_group.int() - 1].squeeze()
        weighted_encodings = torch.matmul(encodings, adjacency_matrices)

        # Trim to embedding dimension
        embedding_dim = self.embedding_dim // 2
        weighted_encodings = weighted_encodings[:, :, :embedding_dim]

        # Apply dense layers
        x = self.dense1(weighted_encodings, space_group.int() - 1)
        x = torch.cat([x.real, x.imag], dim=-1)
        x = F.silu(x)
        x = self.dense2(x)

        return self.norm(x)


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.
        Parameters
        ----------
        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        B, A, _ = atom_in_fea.shape
        _, _, M, _ = nbr_fea.shape

        # Reshape input for original-style computation
        atom_in_fea_reshaped = atom_in_fea.view(B*A, -1)
        nbr_fea_reshaped = nbr_fea.view(B*A, M, -1)
        nbr_fea_idx_reshaped = nbr_fea_idx.view(B*A, M)

        # Original-style convolution
        atom_nbr_fea = atom_in_fea_reshaped[nbr_fea_idx_reshaped, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea_reshaped.unsqueeze(1).expand(B*A, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea_reshaped], dim=2)
        
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(B*A, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea_reshaped + nbr_sumed)

        # Reshape output back to batched format
        out = out.view(B, A, -1)

        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False, graphs_array=None, abc_combinations=None):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.positional_encoding = PositionalEncoding(atom_fea_len, abc_combinations, graphs_array)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, atom_positions, reciprocal_matrices, space_group):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)
        atom_fea = atom_fea + self.positional_encoding(atom_positions, reciprocal_matrices, space_group)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)
