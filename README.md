

# BACQ Benchmark
Implements the framework of the BACQ Benchmark suite in a platform independent way.
At the moment, it provides support for the following QPUs:

* Generic CPU (for reference)
* Dwave 
* (TBA) Pasqal (Ruby machine from GENCI)

And the following scores:

* G-Score (Evaluation of Quantum Optimization heuristics)
* Lin2Score (Evaluation of capability for linear system solving in binary)
* PrimeScore (Evaluation of capability for prime factorisation)

*Reference papers:*
* Louise, S. (2026). G-Score, the Maximum Cardinality Matching Series as a Benchmark for Quantum Computers. In Barbaresco, F., Gerin, F. (eds) Quantum Engineering Sciences and Technologies for Industry and Services. QUEST-IS 2025. Communications in Computer and Information Science, vol. 2743. Springer, Cham. https://doi.org/10.1007/978-3-032-13852-1_26

* Louise, S. (2025). Vector Annealing, a Quantum-Inspired Technique: Benchmarking Performance Against Quantum and Simulated Annealing within the BACQ Framework. In 2025 IEEE International Conference on Quantum Computing and Engineering (QCE), pp. 1989–1996. IEEE, Albuquerque, NM, USA. https://doi.org/10.1109/QCE65121.2025.00217 
 

## 📋 Prerequisites

* Python >= 3.10
* CMake >= 3.18
* Compilateur C++ (GCC/Clang)
* Doxygen 
* Compilateur Latex (pdflatex)
* Pandoc

## 🚀 Installation

* Build & install
```bash
git clone [https://github.com/utilisateur/mon-projet.git](https://github.com/utilisateur/mon-projet.git)
cd BacqBench
mkdir build && cd build
cmake ..
make
```
* Installation as python package (necessary for use with QPUs)
	- inside the main directory "cd .." if required
	- activate your current QPU environment (e.g. DWave Ocean, Pasqal Pulser)
	- install benchmark package using "pip install -e ."
	- N.B. you can do it after running the tests (see below)
Installing in the support environment of your QPU is necessary to run the QPU backend of the benchmark suite.

## Basic unitary tests

```bash
make check 
```

This provides several unitary tests. Not all are implemented yet.

## Usage 
The main python script is BACQBenchmark at the root of the project
The minimum parameter is the target QPU (or CPU comparison)
E.g.
```bash
python3 BACQBenchmark dwave 
```
Will launch the G-Score evaluation on the default architecture of DWave (Advantage/Pegasus ATM).
It is worth noting that this requires the BACQBenchmark to be installed as package in the QPU environement.
If it is not the case, the main program would display "Backend XXX not available"

E.g.
```bash
python3 BACQBenchmark cpu 
```
this command runs the G-Score on Simulated Annealing (CPU backend)

Use 
```bash 
python3 BACQBenchmark -h
```
or 
```bash 
python3 BACQBenchmark --help 
```
To display usage information. Each backend should provide its own help, but in this case it is necessary to provide the backend on the command line.
Being Alpha release, all options may not be implemented yet.

### Choice of the Dwave QPU architecture
To run a benchmark on a Dwave machine, by default, the architecture is set to "pegasus". To specify on which architecture to run, ``-a pegasus`` or ``-a zephyr`` must be added at the end of the command. 

### Details on the lin2score benchmark
To run the lin2score:
E.g. 
```bash 
python3 BACQBenchmark -b lin2score -r 2 -i 2 dwave -a pegasus
```
-r: resolution of the variables
-i: iteration, with the number of variables being i+2. Optional, this value allows to look at only one iteration, without running the entire benchmark experiment. 

### Details on the primescore benchmark
To run the primescore on the D-wave machines:
```bash 
python3 BACQBenchmark -b primescore dwave -p params_primescore.json -a pegasus
```
or
```bash 
python3 BACQBenchmark -b primescore dwave -p params_primescore.json -a zephyr
```

And e.g. *params_primescore.json*: 
```json
{
    "Ns": [91,105],
    "size": [3,4],
    "Method_input": "flux_biases",  
	"Flux_biases_empirical_strength" : 1000,
	"Chain_strength": 2
}
```
Ns                   			  : list of int   - integers to factorize
size                 			  : tuple         - size of the multiplier (number of rows, number of columns)
chain_strength    				  : float         - chain strength, default value is 2.
Method_input       				  : str           - 'flux biases' or 'API'
Flux_biases_empirical_strength    : int           - value of the empirical strength if Method_input is 'flux biases', default value is 1000.
        
*NB: if the backend is not available, the ID of the Dwave machine might have been updated/changed by the provider. This value can be updated in this benchmark in the file hardware\dwave\Dwave_machines.py.* 

## To do 

Here is a preliminary list of pending functionalities:
* Saving log in json 
* Evaluation of benchmark from json results file 


