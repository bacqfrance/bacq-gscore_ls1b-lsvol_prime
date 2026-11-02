#!/bin/python3
 
r""" 
@file CPUBenchmark.py
@package CPUBenchmark is the implementation of function sets for implementing CPU/QPU comparison benchmarks.
Dependencies are:
	- numpy
	- linear algebra for Ising module/library
	- QPUBenchClass library
	- GnBenchLib library
	- argparse python module
	Distributed under EU Public Licence (EUPL) and (c) CEA LIST, QuLIST team
"""
import numpy as np
import sys
import argparse
import random
# add project root to python path
from pathlib import Path
root = Path(__file__).resolve().parents[2]   
sys.path.insert(0, str(root))
# then import from project
from lib.linear_algebra_ising import *
from lib.QPUBenchClass import QPUBenchClassCapability, QPUBenchClass, QPUBenchContext, sampledResults, samplesetClass
from benchmarks.gscore.GnBenchLib import GnQUBO, successProbaGn, gnScore, evalGScore

## helper function to try and load shared C/C++ library independently of the platform (linux/BSD or MacOS or even Windows)
import ctypes
from ctypes import byref, POINTER, c_int
from numpy.ctypeslib import ndpointer
from pathlib import Path
from typing import Optional

def load_library(lib_name: str, relative_path: Optional[Path] = None) -> ctypes.CDLL:
	r"""
	Sideload a library in a multi-plateform way.
	    
	- lib_name : base name (without extension, as the extension is defined by the OS)
	"""
	## -# Define a path if not given 
	if relative_path is None:
		##     - Default behavior : same directory as the current file or in  ./libs/
		CURRENT_FILE = Path(__file__).resolve()
		PROJECT_ROOT = CURRENT_FILE.parents[3]
		lib_path = PROJECT_ROOT / "libs"
		#print(relative_path)
	
	## -# Base OS detection
	##     - case of Windows
	if sys.platform.startswith("win32"): 
		lib_filename = f"{lib_name}.dll"
		#         -On Windows also look in system PATH 
		load_flags = ctypes.RTLD_GLOBAL
	##     - Case of MacOS
	elif sys.platform == "darwin":          
		lib_filename = f"lib{lib_name}.dylib"
		load_flags = ctypes.RTLD_GLOBAL
	##     - default Linux and other Unices
	else:                                   
		lib_filename = f"lib{lib_name}.so"
		load_flags = ctypes.RTLD_GLOBAL

	lib_path = relative_path / lib_filename
	
	try:
		# First try: with explicit path
		if lib_path.exists():
			return ctypes.CDLL(str(lib_path), mode=load_flags)
        
		# Second try: search by name (in PATH/ldconfig)
		print(f"Library {lib_filename} not found in {lib_path}, try by name...")
		return ctypes.CDLL(lib_filename, mode=load_flags)
	
	except OSError as e:
		raise RuntimeError(
			f"Cannot load library '{lib_name}'.\n"
			f"Current detected platform: {sys.platform}\n"
			f"File researched: {lib_filename}\n"
			f"Tentative path: {lib_path}\n"
			f"Error: {e}"
		) from e

## Return library path according to underlying OS.
def get_lib_path(lib_name: str) -> Path:
	## default path is within current directory
	CURRENT_FILE = Path(__file__).resolve()
	base = CURRENT_FILE.parents[2] / "libs"
	#print(base)
	
	## - Case of Windows
	if sys.platform.startswith("win32"): 
		return base / f"{lib_name}.dll"
	## - Case of MacOS
	elif sys.platform == "darwin":       
		return base / f"lib{lib_name}.dylib"
	## - Case of Linux/Unix
	else:                                
		return base / f"lib{lib_name}.so"


## Side load the C++ SA library to use within implementation of the computing with CPU implementation 
## Sideload the library for SA (simpleSA) and prime factorization (factorize_all_c)
lib_path = get_lib_path("BacqBench")

if not lib_path.exists():
	raise FileNotFoundError(f"Library not found: {lib_path}")

lib = ctypes.CDLL(str(lib_path))

#lib_path_prime = get_lib_path("factorize_all")

## CPU Algorithms available for this implementation of reference score
CPUAlgoTypes= {'sa', 'sqa'} ##< 'sa' for simulated annealing, and 'sqa' for simulated quantum annealing (Monte-Carlo simulation)
## N.B.: sqa not yet implemented (Simulated Quantum Annealing)

def selectCPUheuristic(algoType='sa'):
	r""" Select base algorithm for CPU base benchmark. The default is 'sa' (Simulated Annealing)
	"""
	sampler= None
	for algo in CPUAlgoTypes:
		if algoType == algo:
			sampler= algo
	if sampler == None:
		raise ValueError(f"Invalid type of CPU algorithm = {algoType} \n Possible accepted CPU algorithms are {CPUAlgoTypes}")
	print("selected CPU algorithm=", algoType)
	return sampler


## import dependencies 
## using ctypes for importing C/C++ functions
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

# basic definitions
BINARY= 0
SPIN= 1

## signature is: double runSA(int msize, ModelType mtype, double *theJ, double *theh, int *ret)
## it is the same for the MT (openMP) version
lib.runSAmt.argtypes = [
	ctypes.c_int,  # msize (size of the model in number of variables)
	ctypes.c_int,  # mtype (type of mode BINARY (0) or SPIN (1))
	ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"), # theJ
	ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"), # theh
	ctypes.c_int,                                      # nsamp
	ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"), # energies
	ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")    # ret
]

## calling C function runSAmt from Python3
## - input: matrix J (numpy), vector h (numpy), type of variables, number of samples 
## - returns: SA results (set of samples)
def runSAmt_py(J:np.ndarray, h:np.ndarray, mytype: int, nsamp: int):
	""" Python3 caller for C++ function runSAmt """
	## enforce memory contiguity for J and h as C++ parameters
	J= np.ascontiguousarray(J, dtype= np.float64)
	h= np.ascontiguousarray(h, dtype= np.float64)
	
	msize= len(h)
        
	## safety checks (matching sizes)
	if J.shape != (msize,msize):
		raise ValueError("Incorrect parameter J= must be msize*msize with msize is the size of the h vector")
	
	## allocate return values on the python side
	energies= np.empty(nsamp, dtype= np.float64)
	ret= np.empty((nsamp,msize), dtype=np.int32)
	
	## calling the C function
	lib.runSAmt( msize, mytype, J, h, nsamp, energies, ret)
	
	## return samples
	#occvect= np.full_like(energies, 1) ##< This is the number of occurence. This will simplify agregation of results later
	sampleset= tuple(zip(ret, energies)) #, occvect))
	return sampleset


def setGnSARegister(niter, logfile=None, Verbose=False): #-> BinaryQuadraticModel:
	""" Preprocessing for Gn series: set J and h so that runSAmt can use them to solve Gn """
	n= niter ##< n value is passed as the niter (iteration number) of the prepocessing 
	myQ, numval= GnQUBO(n)
	## extract J matrix and h vector from myQ (QUBO matrix of the Gn instance)
	J= np.copy(myQ)
	h= np.zeros(len(J))
	i=0
	for l in J:
		h[i]= l[i] ##< copy diagonal into h vector
		J[i][i]=0  ##< reset diagonal in J matrix to 0 
		i+= 1
	return h, J, (n+1)**3 ##< return values are the h vector, the J matrix and the number of variables



def lowestSamples(samples):
	retlow= 1e9  ## maybe +inf 
	smplow= []
	if len(samples) == 0:
		return []
	## find lowest energy
	minenergy= min(nrg for _, nrg in samples)
	## Gather by vector values
	smpset= {}
	for vec, nrg in samples:
		if nrg != minenergy:
			continue ##< ignore all except lowest energy 
		keyvec= tuple(vec)
		if keyvec in smpset:
			smpset[keyvec].occurences += 1
		else:
			smpset[keyvec]= samplesetClass(vecVal=vec.copy(), energy= minenergy, occurences= 1)
	## prepare output layout for the samples 
	smplow= list(smpset.values())
	smplow.sort(key= lambda r: r.occurences, reverse= True)
	## now, return as a list
	return smplow

def formatSelectSample(sampleset, lowest= True):
	r""" Transform CPU pseudo samples to sampledResults class 
	input: sample set
	output: sampledResults samples
	"""
	#print(sampleset)
	minsamples= lowestSamples(sampleset)
	mysample= sampledResults()
	print("### ->", minsamples)
	mysample.energyCost= minsamples[0].energy 
	print("lowest energy=", mysample.energyCost)
	samplerecord= minsamples
	sampletable= samplerecord[0].vecVal
	totsamples= 0
	lowsamples= 0
	#print(mysample.info)
	for cursample in sampleset:
		#print(cursample)
		totsamples+= 1
		if cursample[1] == mysample.energyCost:
			lowsamples+= +1
	mysample.totNum= totsamples
	mysample.num= lowsamples
	print("total sample num=", totsamples, lowsamples)
	#print(sampletable) #minsamples.variables
	mysample.numQubits= len(minsamples[0].vecVal)
	if mysample.numQubits != len(sampletable):
		raise ValueError("Inconsistency in number of qubits and sample size in formatSelectDwSample")
	mysample.samples= (np.asarray(sampletable),mysample.energyCost, lowsamples)
	# check if only oneliner for returned sampleset
	if mysample.num == 1:
		print(mysample.samples, type(mysample.samples[0]))
		newsamp= mysample.samples[0].reshape(1,-1)
		mysample.samples=(newsamp, mysample.energyCost, lowsamples)
	print("this is my sample=", mysample, "\n", mysample.samples,"-> qbits=", mysample.numQubits)
	smpsel= []
	for smp in mysample.samples:
		smpsel.append((smp))
	
	return mysample

# Function signatures
lib.factorize_all_c.argtypes = [
    c_int,                      # n
    c_int,                      # time_limit_sec
    POINTER(POINTER(c_int)),    # int** out_data
    POINTER(POINTER(c_int)),    # int** out_lengths
    POINTER(c_int),             # int*  out_num_rows
]
lib.factorize_all_c.restype = c_int

lib.free_factorization.argtypes = [POINTER(c_int), POINTER(c_int)]
lib.free_factorization.restype = None

def factorize_all(n: int, time_limit_sec: int = 60) -> list[list[int]]:
    """
    Python bridge to the C++ factorize_all function.
    Returns a list of lists of ints.
    """
    data_ptr    = POINTER(c_int)()      # équivalent à int*
    lengths_ptr = POINTER(c_int)()
    num_rows    = c_int()

    rc = lib.factorize_all_c(
        n,
        time_limit_sec,
        byref(data_ptr),      # → int**
        byref(lengths_ptr),   # → int**
        byref(num_rows)       # → int*
    )
    if rc != 0:
        raise RuntimeError(f"factorize_all_c failed with code {rc}")

    try:
        rows = []
        offset = 0
        for i in range(num_rows.value):
            length = lengths_ptr[i]
            row = [data_ptr[offset + j] for j in range(length)]
            rows.append(row)
            offset += length
        return rows
    finally:
        # Always free the memory allocated on the C++ side
        lib.free_factorization(data_ptr, lengths_ptr)

## Use CPU function using a sieve algorithm to sample the benchmark        
def runFactorizeCPU(nmin, nmax, dum, samplenb):
	mysamp= factorize_all(1000001)
	size= len(mysamp[-1])
	samp= [mysamp[-1],0]
	retsamp= [[samp[0].copy(), samp[1]] for _ in range(samplenb)]
	return retsamp

## dummy preprocess for prime factorization	
def preprocessPrime(n):
	return 55,55,55

## dummy postprocess for prime factorization. We can imagine that the best Top 500 computers could achieve around 55 bits 
## for continuous factorization in the interval [2,n] primescore=log_2 (n) within 24h
def postprocessPrime(ctx):
	return 1, 55


def preprocessLin2(n):
	return 1,1,1

def dummyLin2cpu(nmin,nmax, dum, samplenb):
	samp= [[random.randint(0, 1) for _ in range(500)],0]
	retsamp= [[samp[0].copy(), samp[1]] for _ in range(samplenb)]
	return retsamp

## dummy postprocess for lin1score. We return 1E6 which is a very rough estimate of average CPU capability with usual libraries
def postprocessLin2(ctx):
	return 1, 10000000
	

class CPUBenchmark(QPUBenchClass):
	r""" Concrete implementation of class QPUBenchClass for CPUs
	"""
	capabilitySet= ['qscore', 'gscore', 'lin2score', 'linvol', 'primescore'] ##< set of capabilities for CPUs
	cpuctx= QPUBenchContext() # initialize a QPU context 
	## initialize the remaining as default "uninitialized"
	cpusAlgo= None 
	currentBench= None
	myPreprocess= None
	myCompute= None
	myPostprocess= None
	nIter=0
	preprocessRes= None
	preprocessRes2= None
	computeRes= None
	isInit= False
	loginfile= None
	dat= None
	nqubits= -1
	cpuDispatch= {'gscore':[setGnSARegister, runSAmt_py, evalGScore], 'primescore':[preprocessPrime, runFactorizeCPU, postprocessPrime], 'lin2score':[preprocessLin2, dummyLin2cpu, postprocessLin2]}
	
	@property
	def capabilities(self) -> QPUBenchClassCapability:
		r""" Method to return capabilities of QPU benchmark implementations """
		return QPUBenchClassCapability("CPU", capabilitySet)
	
	@classmethod
	def add_arguments(cls, parser: argparse.ArgumentParser):
		r"""
		This class method provides plateform specific parsing option for command line interface.
		- For CPU it will provide the base algorithm to select (atm, only SA)
		"""
		parser.add_argument('-l', '--logfile', default= 'cpuBench.log', help='output log file (default= cpuBench.log))')
		parser.add_argument('-a', '--alg', default= 'sa', help='Selection of main algorithm for CPU')
	
	
	def is_available(self,qpuArch= None) -> bool:
		r""" Return availability for QPU type
		 - for QPU this is always True"""
		return True

	def selectBenchmark(self, bench: str, num:int, cpualgo= None, **kwargs):
		r""" Select Benchmark (QPU agnostic call, but concrete implementation) """
		print("Select Benchmark=", bench, cpualgo, self.capabilitySet)
		self.cpuctx.currentBench= bench
		## Example: select gscore at n=2
		#print("type=", type(self.capabilitySet), type(bench))
		print("test=", bench in self.capabilitySet)
		print("test2= ", self.capabilitySet[1], bench, "-", len(bench), len(self.capabilitySet[1]))
		if bench in self.capabilitySet:
			self.currentBench= bench
			self.nIter= num
		else:
			print("Warning = None selected")
			return False
		if cpualgo != None:
			print("selecting algorithm from =", cpualgo)
			self.cpusAlgo= cpualgo
		processSet= self.cpuDispatch[bench]
		self.myPreprocess= processSet[0]
		self.myCompute= processSet[1]
		self.myPostprocess= processSet[2]
		#print(processSet[2])
		return True

	def preprocessIteration(self, bench:str, **kwargs) -> bool:
		## Preprocessing: also QPU agnostic, concrete implementation for CPU
		if self.myPreprocess == None:
			return False
		self.preprocessRes, self.preprocessRes2, self.nqubits= self.myPreprocess(self.nIter)
		return True
		 

	def computeIteration(self, **kwargs) -> bool:
		if self.myCompute == None:
			return False
		print("this is algorithm=", self.cpusAlgo)
		self.computeRes= self.myCompute(self.preprocessRes2, self.preprocessRes, BINARY, 500) # 500 samples as default
		self.cpuctx.computeRes= self.computeRes
		self.cpuctx.formatedSamples= formatSelectSample(self.computeRes)
		return True

	def postprocessIteration(self, **kwargs) -> bool:
		if self.myPostprocess == None:
			print("what is happening? ", self.myPostprocess)
			self.postprocessRes= self.myCompute
		else:
			self.postprocessRes= self.myPostprocess(self.cpuctx)
		return True
		
	def __init__(self, iteration, bench, alg, logfile=None):
		loginfile= logfile 
		cpusAlgo= alg
		currentbench= bench
		self.cpuctx.currentBench= bench
		self.cpuctx.nIter= iteration
		self.selectBenchmark( bench, iteration, alg)
		
	def preprocess(self):
		#self.dwsampler= 
		self.cpuctx.nIter= self.nIter
		return self.preprocessIteration(self)
		
		
	def compute(self):
		#self.mysamples= 
		self.computeIteration()
		#self.dwctx.computeRes= self.mysamples
		#print(self.dwctx.computeRes)
		return
	
	def postprocess(self):
		#print(self.myPostprocess, type(self.myPostprocess))
		if self.myPostprocess == None:
			return True, 0
		myret, myscore= self.myPostprocess(self.cpuctx)
		return myret, myscore


if __name__ == "__main__":
	factors = factorize_all(20, time_limit_sec=30)
	for i, f in enumerate(factors, 1):
		print(i, "→", f)

# --------------------------------------------- #

"""
Main program: Implement BACQ scoring suite for DWave familly of QPUs 
"""

#import argparse
#from pathlib import Path
#import os
#import sys
#import json

#parser= argparse.ArgumentParser(prog='dwave-bacqscore', allow_abbrev=True)
#parser.add_argument('-l', '--logfile', default= 'dwaveBench.log', help='output log file (default= dwaveBench.log))')
#parser.add_argument('-s', '--start', type=int, default= 1, help='Start from iteration n=s (default=1) e.g. start G_n at rank s')
#parser.add_argument('-b', '--bench', default= 'gscore', help='Benchmark score to use: default= gscore')
#parser.add_argument('-a', '--arch', default= 'pegasus', help='Selection of architecture type of DWave QPU')

#myargs= parser.parse_args()
#print(myargs)

#logfile= myargs.logfile
#tobench= myargs.bench 
#setarch= myargs.arch 

#print(logfile)
#n= myargs.start
#print(n)

#logf= open(logfile, "w", encoding="utf-8") # overwrite previous if it exists
# # instantiation of QPUBenchClass using the collected data 

#myBench= DWaveBenchmark(1,tobench,setarch, tobench)

#benchdone= False

#while not benchdone:
#	print("############# ?????? nIter=", myBench.nIter)
#	myBench.preprocess()
#	myBench.compute()
#	benchdone, score= myBench.postprocess()
#	myBench.nIter+= 1	

#print("Final score on bench", myBench.currentBench, "is", score)

