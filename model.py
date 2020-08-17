import torch
from torch import nn
from dgl import function as fn

class RGCNLayer(nn.Module):
    def __init__(self, graph,n_in_feats,n_out_feats,activation,dropout,bias, n_rels,n_bases,self_loop):
        super().__init__()
        self.graph = graph
        self.n_in_feats = n_in_feats
        self.n_out_feats = n_out_feats
        self.activation = activation
        self.self_loop = self_loop

        #if dropout:
        self.dropout = nn.Dropout(p=dropout)
        #else:
        #self.dropout = 0

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_out_feats))
        else:
            self.bias = None

        self.n_rels = n_rels
        self.n_bases = n_bases

        if self.n_bases <= 0 or self.n_bases > self.n_rels:
            self.n_bases = self.n_rels

        self.loop_weight = nn.Parameter(torch.Tensor(n_in_feats, n_out_feats))

        # Add basis weights
        self.weight = nn.Parameter(torch.Tensor(self.n_bases, self.n_in_feats, self.n_out_feats))

        if self.n_bases < self.n_rels:
            # Linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.n_rels, self.n_bases))

        self.reset_parameters()

    def reset_parameters(self):
        if self.self_loop:
            nn.init.xavier_uniform_(self.loop_weight)

        nn.init.xavier_uniform_(self.weight)

        if self.n_bases < self.n_rels:
            nn.init.xavier_uniform_(self.w_comp)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def propagate(self, h):
        if self.n_bases < self.n_rels:
            # Generate all weights from bases
            weight = self.weight.view(self.n_bases, self.n_in_feats * self.n_out_feats)
            weight = torch.matmul(self.w_comp, weight).view(self.n_rels, self.n_in_feats, self.n_out_feats)
        else:
            weight = self.weight

        def msg_func(edges):
            w = weight.index_select(dim=0, index=edges.data["type"])
            msg = torch.bmm(edges.src["h"].unsqueeze(1), w).squeeze()
            msg = msg * edges.data["norm"]
            return {"msg": msg}

        self.graph.update_all(msg_func, fn.sum(msg="msg", out="h"))

    def forward(self, h):
        if self.self_loop:

            loop_message = torch.mm(h, self.loop_weight)

            #if self.dropout:
            loop_message = self.dropout(loop_message)

        self.graph.ndata["h"] = h

        # Send messages through all edges and update all nodes
        self.propagate(h)

        h = self.graph.ndata.pop("h")

        if self.self_loop:
            h = h + loop_message

        if self.bias is not None:
            h = h + self.bias

        if self.activation:
            h = self.activation(h)

        return h

class RGCN(nn.Module):
    """
    Relational Graph Convolutional Network for entity classification.

    Parameters
    ----------
    graph: dgl.DGLGraph
        The graph on which the model is applied.
    features: torch.FloatTensor
        Feature matrix of size n_nodes * n_in_feats.
    n_hidden_feats: int
        The number of features for the input and hidden layers.
    n_hidden_layers: int
        The number of hidden layers.
    activation: torch.nn.functional
        The activation function used by the input and hidden layers.
    dropout: float
        The dropout rate.
    n_rels: int
        The number of relations in the graph.
    n_bases: int
        The number of bases used by the model.
    self_loop: boolean
        Use self-loop in the model

    References
    ----------
    M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, and M. Welling,
    “Modeling Relational Data with Graph Convolutional Networks,” arXiv:1703.06103 [cs, stat], Mar. 2017.
    """

    def __init__(self,graph,features,n_hidden_feats,n_hidden_layers,
                 n_classes,activation,dropout,n_rels,n_bases,self_loop,
                 click_lstm_hidden_size,click_sequence_length,
                 click_lstm_num_layers):
        super().__init__()
        self.features = features
        #self.click_function_type = click_function_type
        n_in_feats = features.size(1)
        self.click_sequence_length =click_sequence_length
        self.layers = nn.ModuleList()
        #if self.click_function_type=='LSTM':
        self.click_function =nn.LSTM(20, click_lstm_hidden_size,
                                         num_layers = click_lstm_num_layers, batch_first=True,dropout=dropout)

        self.layers.append(
            RGCNLayer(graph=graph,n_in_feats=n_in_feats,n_out_feats=n_hidden_feats,activation=activation,
                      dropout=0,bias=True,n_rels=n_rels,n_bases=n_bases,self_loop=self_loop)
        )

        # Hidden layers
        for _ in range(n_hidden_layers):
            self.layers.append(
                RGCNLayer(graph=graph,n_in_feats=n_hidden_feats,n_out_feats=n_hidden_feats,activation=activation,dropout=dropout,
                    bias=True,n_rels=n_rels,n_bases=n_bases,self_loop=self_loop
                )
            )

        # # Output layer
        self.output_layer = nn.Linear(click_lstm_hidden_size+2*n_hidden_feats, n_classes)

    def forward(self, x, click_data, edge_src, edge_dst, truncate_sequence_lstm):
        """
        Defines how the model is run, from input to output.

        Parameters
        ----------
        x: torch.FloatTensor
            (Input) feature matrix of size n_nodes * n_in_feats.

        Return
        ------
        h: torch.FloatTensor
            Output matrix of size n_nodes * n_classes.
        """

        h = x
        for layer in self.layers:
            h = layer(h)

        relevant_src_nodes_embes = h[edge_src]
        relevant_dst_nodes_embes = h[edge_dst]
        output_click, hidden = self.click_function(click_data.view(click_data.size(0),self.click_sequence_length,20))
        output_click = torch.squeeze(output_click[:,truncate_sequence_lstm,:])

        if len(output_click.size())==1:
            output_click=torch.unsqueeze(output_click,0)

        final_node_edge_embed = torch.cat((relevant_src_nodes_embes,relevant_dst_nodes_embes,output_click),axis=1)

        output = self.output_layer(final_node_edge_embed)
        return output
