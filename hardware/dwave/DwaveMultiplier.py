import numpy as np
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import json
from fractions import Fraction
import networkx as nx
from dwave.system import DWaveSampler
from pathlib import Path

class Multiplier():
    ## Definition of the Multiplier parameters
    def __init__(self, N, Dwave_Machine, sizes=None, chain_strength=2):
        r'''
        :param N                : int   - integer to factorize
        :param Dwave_Machine    : dict  - dictionnary of properties of the dwave annealer considered
        :param sizes            : tuple - size of the multiplier (number of rows, number of colums), 
        :param chain_strength   : float - chain strength, default value is 2.
        '''
        self.N = N
        if sizes==None:
            self.binary_N = np.binary_repr(N)
            self.n = len(self.binary_N)
            self.sizes = (self.n//2+1,self.n-1)
        else: 
            self.sizes = sizes
        self.binary_N = np.binary_repr(N, self.sizes[0]+self.sizes[1])
        self.n = len(self.binary_N)
        self.n_row = self.sizes[0]
        self.n_col = self.sizes[1]

        self.inputs_to_fix = {'0':[], '1': []}
        self.factors_nodes = {'in1':{i : [] for i in range(self.n_col)}, 'enable':{i : [] for i in range(self.n_row)}}
        self.Dwave_Machine = Dwave_Machine

        self.H_range = self.Dwave_Machine['H_range']
        self.J_range = self.Dwave_Machine['J_range']

        ## -# Load the Ising hamiltonian of the CFA
        BaseDir= Path(__file__).resolve().parent
        with open(f'{BaseDir}/CFAs-Library/CFA_{self.Dwave_Machine['CFA_tile']}.json', 'r') as f:
            data = json.load(f)
        self.biases, self.couplings, self.offset, self.cfa_plmt = self.read_data(data)
        
        self.chain_strength = chain_strength
        
        self.Compatibility = None

        ## -# Define the coordinates and subgraphs depending on the Dwave topology.
        if self.Dwave_Machine['CFA_tile'] == 'Zephyr':
            self.topologie_graph = dnx.zephyr_graph(m=self.Dwave_Machine['m'])
            self.coords = dnx.zephyr_coordinates(m=self.Dwave_Machine['m'])
            self.pos = dnx.zephyr_layout(self.topologie_graph)

        elif self.Dwave_Machine['CFA_tile'] == 'Pegasus':
            self.topologie_graph = dnx.pegasus_graph(m=self.Dwave_Machine['m'])
            self.coords = dnx.pegasus_coordinates(m=self.Dwave_Machine['m'])
            self.pos = dnx.pegasus_layout(self.topologie_graph)

        else:
                    raise ValueError('CFA_tile undefined')
    def make_nodes(self,row, col, m, hv, p):
            r'''
            :param row: int - index of the row 
            :param col: int - index of the column 
            :param m  : int - parameter of the Dwave machine
            :param hv : int - horizontal or vertical parameter of the nice coordinates of Pegasus topology
            :param p  : int - position parameter of the nice coordinates of Pegasus topology
            '''
            t = col % 3
            v = col // 3
            u = row+ v
            return self.coords.nice_to_linear((2 - t, u, m - 3 - v, hv, p))
    def cfa_nodes(self, row, col):
        r'''
        :param row: int - index of the row 
        :param col: int - index of the column 
        '''
        if self.Dwave_Machine['CFA_tile'] == 'Pegasus':
            nodes  = [self.make_nodes(row, col, self.Dwave_Machine['m'], hv, p) for hv in [0,1] for p in [0,1,2,3]]
            nodes += [self.make_nodes(row+1, col-1, self.Dwave_Machine['m'], 0, 0)]
            dict = {i:n for n,i in enumerate(nodes)}
        elif self.Dwave_Machine['CFA_tile'] == 'Zephyr':
            m = self.Dwave_Machine['m']
            nodes  = [self.coords.zephyr_to_linear((0, m*2-2-col , 2, ((row)%2), (row//2)))]
            nodes += [self.coords.zephyr_to_linear((1, row, k, -((col+1)%2), m*2-1-(col+1)//2))for k in [4,5,6,7]]
            nodes += [self.coords.zephyr_to_linear((0, m*2-2-col , k, ((row)%2), (row//2))) for k in [4,5,7]]
            nodes += [self.coords.zephyr_to_linear((1, row, 5, -((col)%2), m*2-1-(col//2)))]
            nodes += [self.coords.zephyr_to_linear((0, m*2-2-col , 6, +((row+1)%2), +((row+1)//2)))]
            dict = {i:n for n,i in enumerate(nodes)}
        return dict
  
    def make_subgraph(self, row, col):
        r'''
        :param row: int - index of the row 
        :param col: int - index of the column 
        '''
        nodes = self.cfa_nodes(row, col).keys()
        return self.topologie_graph.subgraph(nodes)
    
    ## -# Reads the biases, couplings, offset and CFA indices from the corresponding .json in the library.
    def read_data(self, data):
        r'''
        :param data: 
        '''
        def to_float(x):
            return float(Fraction(x))
        bqm_raw = data["bqm"]
        cfa_plmt = data['plmt']
        offset = to_float(bqm_raw["offset"])
        biases = {}
        couplings = {}
        for k, v in bqm_raw.items():
            if k in ["g_min", "offset"]:
                continue
            if k.startswith("("):  # J_ij
                i, j = eval(k)
                couplings[(int(i), int(j))] = to_float(v)
            else:  # h_i
                biases[int(k)] = to_float(v)
        return biases, couplings, offset, cfa_plmt
    
    ## -# Map the a CFA, for a given row and column of the multiplier, onto the topology/coordinates of the Dwave Machine.
    def cfa_tile_to_graph(self, row, col, tile_0=(0,0)):
        ''' 
        :params row: int - row number of the CFA in the multiplier
        :params col: int - column number of the CFA in the multiplier
        :params tile_0: tuple - Position of the first CFA on the real machine
        '''
        cfa_tile = self.cfa_nodes(row + tile_0[0], col + tile_0[1])
        
        if row==0: # First row : in2=0 and enable_0 = 1
            in2_idx = self.cfa_plmt['in2']
            enable_0_idx = self.cfa_plmt['enable'] 
            self.inputs_to_fix['0'].append(list(cfa_tile.keys())[in2_idx])
            self.inputs_to_fix['1'].append(list(cfa_tile.keys())[enable_0_idx])

        if col==0:   # First column: cin=0, in1_0 = 1 and out is known
            cin_idx = self.cfa_plmt['cin']
            in1_idx = self.cfa_plmt['in1']
            out_idx = self.cfa_plmt['out']
            self.inputs_to_fix['0'].append(list(cfa_tile.keys())[cin_idx])
            self.inputs_to_fix['1'].append(list(cfa_tile.keys())[in1_idx])
            self.inputs_to_fix[self.binary_N[self.n-row-1]].append(list(cfa_tile.keys())[out_idx])

        if row==self.n_row-1 and col>0: # Last row, out is known
            out_idx = self.cfa_plmt['out']
            self.inputs_to_fix[self.binary_N[self.n-row-col-1]].append(list(cfa_tile.keys())[out_idx])

            if col ==self.n_col-1: # On for the CFA of the last row and last column, the value of cout is known too.
                cout_idx = self.cfa_plmt['cout']
                self.inputs_to_fix[self.binary_N[0]].append(list(cfa_tile.keys())[cout_idx])

        Graph_cfa_biases = {node : self.biases[i] for node, i in cfa_tile.items()}
        Graph_cfa_couplings= {}
        for (u,v),weight in self.couplings.items():
            Graph_cfa_couplings.update({(list(cfa_tile.keys())[u],list(cfa_tile.keys())[v]):weight})

        return Graph_cfa_biases, Graph_cfa_couplings
    
    ## -# Build the complete graph of the multiplier onto the topology of the Dwave Machine.
    def build_graph_complete(self, tile_0=(0,0)):
        r'''
        :param tile_0: tuple - position of the first CFA of the multiplier on the Dwave Machine topology. default value is (0,0)
        '''
        self.tile_0 = tile_0
        graph_complete_biaises = {}
        graph_complete_couplings = {}
        chains={}
        offset = 0
        for row in range(self.n_row):
            for col in range(self.n_col):
                cfa_graph = self.cfa_tile_to_graph(row, col, tile_0)
                ## -# Build the chains between CFAs
                if row >0:
                    if col < self.n_col-1:
                        ## -# in2-out : Shared qubit. 
                        in2_idx = self.cfa_plmt['in2']
                        node = list(cfa_graph[0].keys())[in2_idx]
                        prev_out =  graph_complete_biaises[node]
                        now_in2 = cfa_graph[0][node]
                        if not self.H_range[0] < prev_out + now_in2 <self.H_range[1]:
                            raise ValueError(f'Error line {row}, colomn {col} qubit sharing out-in2: {prev_out + now_in2}')
                        cfa_graph[0][node] += prev_out
      
                    else:
                        ## -# in2-cout : chain(s), depending on the topology. 
                        if self.Dwave_Machine['CFA_tile']=='Zephyr': 
                            in2_idx = self.cfa_plmt['in2']
                            cout_idx = self.cfa_plmt['cout']
                            chains_nodes = (list(cfa_graph[0].keys())[in2_idx],list(self.cfa_tile_to_graph(row-1,col, tile_0 )[0].keys())[cout_idx])
                            chains.update({chains_nodes:-self.chain_strength})
                            offset +=self.chain_strength
                        elif self.Dwave_Machine['CFA_tile']=='Pegasus':
                            chain_cout_in2 =[(
                            self.make_nodes(row-1+tile_0[0], col+tile_0[1], self.Dwave_Machine['m'], 1, 2),self.make_nodes(row-1+tile_0[0], col+1+tile_0[1], self.Dwave_Machine['m'], 0, 3)),
                            (self.make_nodes(row-1+tile_0[0], col+1+tile_0[1], self.Dwave_Machine['m'], 0, 3),self.make_nodes(row-1+tile_0[0], col+1+tile_0[1], self.Dwave_Machine['m'], 1, 3)),
                            (self.make_nodes(row-1+tile_0[0], col+1+tile_0[1], self.Dwave_Machine['m'], 1, 3), self.make_nodes(row+tile_0[0], col+tile_0[1], self.Dwave_Machine['m'], 0, 0))]
                            for edge in chain_cout_in2:
                                chains.update({edge:-self.chain_strength})
                                offset +=self.chain_strength
                        
                    if col>0:
                        ## -# in1-in1 chain
                        in1_idx = self.cfa_plmt['in1']
                        chains_nodes = (list(cfa_graph[0].keys())[in1_idx],list(self.cfa_tile_to_graph(row-1,col, tile_0)[0].keys())[in1_idx])
                        chains.update({chains_nodes:-self.chain_strength})
                        offset += self.chain_strength

                        ## -# enable-enable chain
                        if self.Dwave_Machine['CFA_tile']=='Zephyr':    
                            enable_idx = self.cfa_plmt['enable']
                            chains_nodes = (list(cfa_graph[0].keys())[enable_idx],list(self.cfa_tile_to_graph(row,col-1, tile_0)[0].keys())[enable_idx]) #chain enable=enable
                        elif self.Dwave_Machine['CFA_tile']=='Pegasus':
                            enable_idx = self.cfa_plmt['enable']
                            enable2_idx = self.cfa_plmt['enable_2']
                            chains_nodes = (list(cfa_graph[0].keys())[enable_idx],list(self.cfa_tile_to_graph(row,col-1, tile_0)[0].keys())[enable2_idx]) #chain enable=enable
                        chains.update({chains_nodes:-self.chain_strength})
                        offset +=self.chain_strength

                if col>0:
                    ## -# cin-cout shared qubit
                    if self.Dwave_Machine['CFA_tile']=='Zephyr': 
                        cin_idx = self.cfa_plmt['cin']
                        node = list(cfa_graph[0].keys())[cin_idx]
                        prev_cout =  graph_complete_biaises[node]
                        now_cin = cfa_graph[0][node]
                        if not self.H_range[0] < prev_cout + now_cin < self.H_range[1]:
                            raise ValueError(f'Error line {row}, colomn {col} qubit sharing cout-cin: {prev_cout + now_cin} ')
                        cfa_graph[0][node] += prev_cout
                    ## -# cin-cout chain
                    elif self.Dwave_Machine['CFA_tile']=='Pegasus':
                        cin_id = self.cfa_plmt['cin']
                        cout_id = self.cfa_plmt['cout']
                        chains_nodes = (list(cfa_graph[0].keys())[cin_id],list(self.cfa_tile_to_graph(row,col-1, tile_0)[0].keys())[cout_id]) #chain cout_id=cout_id
                        chains.update({chains_nodes:-self.chain_strength})
                        offset +=self.chain_strength

                graph_complete_biaises.update(cfa_graph[0])
                graph_complete_couplings.update(cfa_graph[1])
                offset+= self.offset

                ## -# Keep track of the qubits corresponding to the factors A(in1) and B(enable). 
                self.factors_nodes['in1'][col].append(list(cfa_graph[0].keys())[self.cfa_plmt['in1']])
                self.factors_nodes['enable'][row].append(list(cfa_graph[0].keys())[self.cfa_plmt['enable']])

        ## -# Qubit values to impose: first bits of A (in1_0) and B (enable_0) and values of out=N_i and cin=0 and in2=0 fist column and first row.
        self.inputs_to_fix['0'] = list(set(self.inputs_to_fix['0']))
        self.inputs_to_fix['1'] = list(set(self.inputs_to_fix['1']))

        filtered_couplings = {edge: coupling for edge, coupling in graph_complete_couplings.items() if abs(coupling) > 0}
        return graph_complete_biaises, filtered_couplings, chains, offset
    
    ## -# Building of the graph of the Multiplier
    def plot_multiplier(self, combined_graph=None, tile_0=(0,0), plot = False, plot_inputs = False):
        r'''
        :param combined_graph: dict - Bias, couplings, chains and offset of the multiplier graph. result of the function self.build_graph_complete.
        :param tile_0: Description
        :param plot: Bool - True: plots the multiplier graph onto the machine topology, chains in red. 
        :param plot_inputs: Bool - True: if plot, identifies the qubits' values to fix, purple for 0, green for 1.
        '''
        if combined_graph is None :
            combined_graph = self.build_graph_complete(tile_0)
        G = nx.Graph()
        for node, bias in combined_graph[0].items():
            if abs(bias)>0:
                G.add_nodes_from([node])
        for edge, coupling in combined_graph[1].items(): 
            if abs(coupling)>0:
                G.add_edges_from([edge])    
        G.add_edges_from(combined_graph[2])
        Chains= nx.Graph()
        Chains.add_edges_from(combined_graph[2])

        if plot:
            fig, ax = plt.subplots(figsize=(18, 18))
            
            nx.draw_networkx_nodes(self.topologie_graph, pos=self.pos, node_color='lightgray', node_size=10, ax=ax)
            nx.draw_networkx_edges(self.topologie_graph, pos=self.pos, edge_color='lightgray', ax=ax, width=0.5)
            nx.draw_networkx_nodes(G, self.pos, node_size=12, node_color='C0')
            nx.draw_networkx_edges(G, self.pos, width=0.5, edge_color='C0', alpha=0.8)
            nx.draw_networkx_edges(Chains, pos=self.pos, edge_color='C3', width=2, ax=ax)
            plt.margins(-0.02)
            if plot_inputs: 
                inputs_0 = nx.Graph()
                inputs_0.add_nodes_from(self.inputs_to_fix['0'])
                inputs_1 = nx.Graph()
                inputs_1.add_nodes_from(self.inputs_to_fix['1'])
                nx.draw_networkx_nodes(inputs_0, self.pos, node_size=12, node_color='C4')
                nx.draw_networkx_nodes(inputs_1, self.pos, node_size=12, node_color='C8')
        return G, Chains, combined_graph
    
    ## -# test the compatibility of the Mutliplier graph and the available qubits and couplings of the real Dwave machine.
    def Compatibility_w_Real_Graph(self, G, Chains, plot_inputs=True):
        '''
        :param G: Graph - Multiplier's graph defined in self.plot_multiplier() (without the chains)
        :param Chains: Graph - Chains's graph defined in self.plot_multiplier()
        :param plot_inputs: Bool - True: if plot, identifies the qubits' values to fix, purple for 0, green for 1.
        '''
        def inverse(edge):
            (u,v) = edge
            return (v,u)
        ## -# Available qubits and couplings of the real Dwave machine.
        qpu_graph = DWaveSampler(solver={'graph_id': self.Dwave_Machine['QPU_ID']}).to_networkx_graph()
        plt.figure(figsize=(18,18))  
        nx.draw_networkx_nodes(qpu_graph, pos=self.pos, node_color='lightgray', node_size=10)
        nx.draw_networkx_edges(qpu_graph, pos=self.pos, edge_color='lightgray',  width=0.5)    

        ## -# Identifies if some nodes are not working
        nodes_incl = [node in list(qpu_graph.nodes()) for node in list(G.nodes())]
        if all(nodes_incl):
            non_working_nodes = []
            print('All the selected nodes are working')
            self.Compatibility = True
        else: 
            non_working_nodes = [node  for i, node in enumerate(list(G.nodes())) if not nodes_incl[i]]
            print('Non-working nodes are:', non_working_nodes)
            self.Compatibility = False
        ## -# Identifies if some edges are not working    
        edges_incl = [edge in list(qpu_graph.edges()) or inverse(edge) in list(qpu_graph.edges()) for edge in list(G.edges()) ]
        if all(edges_incl):
            non_working_edges = []
            print('All the selected edges are working')
        else: 
            non_working_edges = [edge  for i, edge in enumerate(list(G.edges())) if not edges_incl[i]]
            print('Non-working edges are:', non_working_edges)
            self.Compatibility = False

        nx.draw_networkx_nodes(G, self.pos, node_size=12, node_color='C0')
        nx.draw_networkx_edges(G, self.pos, width=0.5, edge_color='C0', alpha=0.8)
        nx.draw_networkx_edges(Chains, pos=self.pos, edge_color='C0', width=0.5)

        non_working_edges_graph = nx.Graph()
        non_working_edges_graph.add_edges_from(non_working_edges)
        non_working_nodes_graph = nx.Graph()
        non_working_nodes_graph.add_nodes_from(non_working_nodes)

        ## -# Non-working nodes in orange, non-working edges in red. 
        nx.draw_networkx_nodes(non_working_nodes_graph, self.pos, node_size=12, node_color='C1')
        nx.draw_networkx_edges(non_working_edges_graph, self.pos, width=0.5, edge_color='C3', alpha=0.8)

        if plot_inputs:
            inputs_0 = nx.Graph()
            inputs_0.add_nodes_from(self.inputs_to_fix['0'])
            inputs_1 = nx.Graph()
            inputs_1.add_nodes_from(self.inputs_to_fix['1'])
            nx.draw_networkx_nodes(inputs_0, self.pos, node_size=12, node_color='C4')
            nx.draw_networkx_nodes(inputs_1, self.pos, node_size=12, node_color='C8')
        plt.title(self.Dwave_Machine['name'])
        plt.margins(-0.02)









                                    
