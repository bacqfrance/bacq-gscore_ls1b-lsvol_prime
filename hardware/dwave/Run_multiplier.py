import os
import math
from dimod import BinaryQuadraticModel

from .Dwave_machines import Dwave_Machines_dict
from .DwaveMultiplier import *
#from Save_Results import *

def save_dwave_txt(sampleset, filepath, title="RESULTS"):
    # Extract samples
    samples = [
        {
            "energy": float(s.energy),
            "num_occurrences": int(s.num_occurrences)
        }
        for s in sampleset.data(["energy", "num_occurrences"])
    ]

    if not samples:
        print("No results to save.")
        return

    # Total shots and ground state
    total_shots = sum(s["num_occurrences"] for s in samples)
    ground_energy = min(s["energy"] for s in samples)

    ground_count = sum(
        s["num_occurrences"]
        for s in samples
        if round(s["energy"], 6) == round(ground_energy, 6)
    )

    success_rate = 100 * ground_count / total_shots

    # Group results by energy
    energy_counts = {}
    for s in samples:
        energy = round(s["energy"], 6)
        energy_counts[energy] = (
            energy_counts.get(energy, 0) + s["num_occurrences"]
        )

    sorted_energies = sorted(energy_counts)
    
    # Write TXT file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(f"{title:^50}\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Success : {success_rate:.2f}%".replace(".", ",") + "\n")
        f.write(f"Ground state : {ground_energy}\n")
        f.write(f"Ground state count : {ground_count}/{total_shots}\n\n")

        f.write("=" * 50 + "\n")
        f.write("ENERGY LEVELS\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"{'Level':<10}{'Energy':<18}{'Count':<12}{'Probability':<15}\n")
        f.write("-" * 55 + "\n")

        for level, energy in enumerate(sorted_energies, 1):
            count = energy_counts[energy]
            probability = 100 * count / total_shots

            f.write(
                f"{level:<10}{energy:<18.6f}{count:<12}"
                f"{probability:.2f}%\n".replace(".", ",")
            )

    print(f"Success : {success_rate:.2f}%".replace(".", ","))
    print(f"Text file → {filepath}")
    
def returnDwavePrime(ctx):
	fact= ctx.target
	score= math.log(fact,2)
	return 1, score
	
def preDwaveMultiplier(sampler, ni, ctx):
	#print(sampler, "\n..\n", ni, "\n..\n",ctx)
	return ctx, ni, 12

def useDwaveMultiplier(sampler, ni, ctx):
    r"""
        Function to run the benchmark on specified integer
        Input:
            - arch: either pegasus or zephyr 
            - toFactorize: list of number to factorize 
            - This function will hide several details from the Dwave implementation and are ATM quite dependent on the actual layout of the QPU (it targets very specific QPUs using their working qubits and working links)
    """
    arch = ctx.arch
    params = ctx.params
    toFactorize = params['Ns']
    size = tuple(params['size'])
    Method_input = params['Method_input']
    Empirical_strength = params['Flux_biases_empirical_strength']
    chain_strength = params['Chain_strength']


    myarch= {'pegasus':'Advantage_', 'zephyr':'Advantage2_'} # name partern associated to the QPU architectures
	
    myBaseName=  [myarch[arch]]
    print(myBaseName)
    # verify
    assert(len(myBaseName) == 1)
    # select QPUs
    myQPU= {
        k:v
        for k,v in Dwave_Machines_dict.items()
        if myBaseName[0] in v.get("name","")
    }
    # verify
    assert(len(myQPU)>0)
    if ni> 1:
        toFactorize= [ni]
        ctx.target= ni

    for k,dw in myQPU.items():
        print(k, dw, "\n", dw["posm"])
        dwsamples= Multiplier_to_Run(toFactorize, dw["posm"], size, k, ctx, Method_input=Method_input, chain_strength=chain_strength, Empirical_strength=Empirical_strength)
    # if arch == 'zephyr':
    # 	k, val= next(iter(myQPU.items()))
    # 	print(k, val)
    # 	dwsamples= Multiplier_to_Run(toFactorize, val["posm"], (3,4), k, ctx, Method_input='API')
    # end
    return ctx.computeRes

## -# Run the Multiplier on the Dwave Machine
class Multiplier_to_Run(): 
    def __init__(self, Ns, start_tile, sizes, id_Dwave_machine, dwcontext, chain_strength =2,Folder_result_name='_Results', Method_input = 'flux_biases', Empirical_strength = 1000):
        r'''
        :param Ns                   : list of int   - integers to factorize
        :param start_tile           : tuple         - position of the first CFA onto the topology of the Dwave machine
        :param sizes                : tuple         - size of the multiplier (number of rows, number of columns)
        :param id_Dwave_machine     : int           - number of the Dwave machine in the dictionnary defined in Dwave_machines.py
        :param chain_strength       : float         - chain strength, default value is 2.
        :param Folder_result_name   : str           - name of the folder for the results.
        :param Method_input         : str           - 'flux biases' or 'API'
        :param Empirical_strength   : int           - value of the empirical strength if Method_input is 'flux biases', default value is 1000.
        '''
        self.Empirical_strength = Empirical_strength

        ## -# First integer to fatorize - check the compatibility with the real machine: 
        Dwave_Machine = Dwave_Machines_dict[id_Dwave_machine]

        N = Ns[0]
        Mult = Multiplier(N, Dwave_Machine=Dwave_Machine, sizes=sizes, chain_strength=chain_strength)
        Ising_dict = Mult.build_graph_complete(start_tile)
        G, Chains, combined_graph = Mult.plot_multiplier(combined_graph=Ising_dict, plot=True, plot_inputs=True)
        Mult.Compatibility_w_Real_Graph(G, Chains, plot_inputs=True)
        if not Mult.Compatibility:
            raise ValueError('Mutliplier graph incompatible with the machine typology')
        
        else: 
            ## -# if Multiplier compatible with the machine, run all the factorization problems. 
            for N in Ns: 
                Mult = Multiplier(N, Dwave_Machine=Dwave_Machine, sizes=sizes, chain_strength=chain_strength)
                Ising_dict = Mult.build_graph_complete(start_tile)
                G, Chains, combined_graph = Mult.plot_multiplier(combined_graph=Ising_dict, plot=True, plot_inputs=True)

                ## -# Prepare the problem for the real machine
                qpu = DWaveSampler(solver={'graph_id': Dwave_Machine['QPU_ID']})
                bqm = BinaryQuadraticModel.from_ising(h=Ising_dict[0], J={**Ising_dict[1], **Ising_dict[2]}, offset= Ising_dict[3])
                params = {
                    "num_reads":    1000,
                    "annealing_time": 10,
                }

                ## -# Definition of the inputs 
                if Method_input =='API': 
                    bqm.fix_variables({node: -1 for node in Mult.inputs_to_fix['0']})
                    bqm.fix_variables({node: 1 for node in Mult.inputs_to_fix['1']})

                if Method_input == 'flux_biases':
                    def _choose_flux_bias_for_qbts(qbt_values, strength=self.Empirical_strength):
                        input_qbts_flux_bias = {qbt: spin * Dwave_Machine['PHI0']  * strength for qbt, spin in qbt_values.items()}
                        fb_list = [0] * Dwave_Machine['#qubits']
                        for qbt, fb in input_qbts_flux_bias.items():
                            fb_list[qbt] = fb
                        return fb_list
                    
                    dict_spin_flux = {**{node: -1 for node in Mult.inputs_to_fix['0']},**{node: 1 for node in Mult.inputs_to_fix['1']}}
                    params['flux_biases'] = _choose_flux_bias_for_qbts(dict_spin_flux, strength= Empirical_strength)
                    params['flux_drift_compensation'] = False

                ## -# Check of the reduced ising
                h_reduced, J_reduced, offset = bqm.to_ising()
                assert all(Dwave_Machine['J_range'][0] <= v <= Dwave_Machine['J_range'][1] for v in J_reduced.values()), "J hors range"
                assert all(Dwave_Machine['H_range'][0] <= v <= Dwave_Machine['H_range'][1] for v in h_reduced.values()), "h hors range"
                
                ## -# Run and save the results 
                sampleset_qpu = qpu.sample(bqm,**params)
                folder_result = Dwave_Machine['name'] + Folder_result_name
                os.makedirs(folder_result, exist_ok=True)
                fp = f"{folder_result}/N{N}_Ta_{params['annealing_time']}_cs{Mult.chain_strength}_A{Mult.n_row}_B{Mult.n_col}_{Method_input}".replace(".", "p")

                save_dwave_txt(sampleset_qpu, fp+'.txt', title=f"RESULTS_{Dwave_Machine['name']}_N{Mult.N}")
                #print(sampleset_qpu)
                dwcontext.computeRes= sampleset_qpu
                
if __name__ == "__main__":
	useDwaveMultiplier('zephyr')

