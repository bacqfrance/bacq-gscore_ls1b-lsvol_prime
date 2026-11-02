#!/bin/python3
 
## @file DWaveBenchmark.py
## @package DWaveBenchmark 
## @brief This is the implementation of function sets for implementing DWave QPU particular access (clean Ocean SDK installed).
## Dependencies are:
##	- numpy
##	- linear algebra for Ising module/library
##	- QPUBenchClass library
##	- GnBenchLib library
##	- LinBench library
##	- Dwave libraries (Ocean Toolkit)
## Distributed under EU Public Licence (EUPL) 1.2 and (c) CEA LIST, QuLIST team
## Initial release 0.1 (not feature complete) 2026Q4

import numpy as np
from lib.linear_algebra_ising import *
from lib.QPUBenchClass import QPUBenchClassCapability, QPUBenchClass, QPUBenchContext, sampledResults
#from benchmarks.gscore import GnBenchLib
from benchmarks.gscore.GnBenchLib import GnQUBO, successProbaGn, gnScore, evalGScore
from benchmarks.linscore.LinBench2 import *

## DWave Available Architectures= make it evolve through the commercial availability of QPUs
DWaveArchTypes= {'pegasus', 'zephyr'} # N.B.: chimera is not available anymore (obsolete)

## import dependencies from dimod, ocean and dwave python libraries
from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler
from dwave.system import EmbeddingComposite, FixedEmbeddingComposite
from dimod import concatenate, to_networkx_graph
from minorminer import find_embedding

## elements for factorization
from .Run_multiplier import preDwaveMultiplier, useDwaveMultiplier, returnDwavePrime

## Select DWave QPU by using architecture and requiring an available QPU with the given architecture
def selectDWaveQPU(archType='pegasus'):
	sampler= None
	for arch in DWaveArchTypes:
		if archType == arch:
			sampler= DWaveSampler(solver=dict(topology__type=arch))
	if sampler == None:
		raise ValueError(f"Invalid type of DWave architecture= {archType} \n Possible accepted architectures are {DWaveArchTypes}")
	print("selected DWave sampler QPU arch=", archType)
	return sampler


def generate_embedding(bqm: BinaryQuadraticModel, sampler: DWaveSampler, seeds: list[int], return_num_qubits: bool=False):
	r"""
	Generate embeddings for the bqm on the sample with the given seeds. 
	Returns the one with the minimum number of qubits.
	If return_num_qubits=True it returns the best_embedding the number of qubits in the best embedding
	else returns the best embedding
	"""
	embeddings = []
	sampler_graph = sampler.to_networkx_graph()
	bqm_graph = to_networkx_graph(bqm)
	## generate embeddings using a series of random seeds and the standard embedding heuristic from DWave
	for seed in seeds:
		emb = find_embedding(bqm_graph, sampler_graph, random_seed=seed)
		embeddings.append(emb)
	## find the one with the lowest number of qubits= lowest number of qubits usually correlates with better results
	best_emb_num_qubits = 999999
	best_emb = None
	best_seed = None
	## select the best embedding wrt the number of qubits for the problem
	for emb in embeddings:
		## Count the number of qubits in this embedding
		num_qubits = 0
		for chain in emb.values():
			num_qubits += len(chain)
			
			if num_qubits and num_qubits < best_emb_num_qubits:
				best_emb_num_qubits = num_qubits
				best_emb = emb
                
	if best_emb == None:
		print("NO EMBEDDING WAS FOUND")
	if return_num_qubits:
		return best_emb, best_emb_num_qubits
	return best_emb

## setGnDwaveRegister set the DWave register to solve the Gn problem 
## - Inputs are:
## 	- sampler (DWave QPU ID)
##	- niter, iteration number 
##	- logfile (N.B.: not yet implemented TBA)
##	- verbose flag (default False)
## - Output: model (ready to run on QPU), associated minor-embedding, and number of qubits
def setGnDwaveRegister(mysampler, niter, ctx, logfile=None, Verbose=False)-> BinaryQuadraticModel:
	## -# Prepare model by calling GnQUBO function from GnBenchLib
	n= niter
	myQ, numval= GnQUBO(n)
	M= len(myQ)
	N= M
	h= getHQUBOforDimod(myQ)
	J= getJforDimod(myQ)
	model = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.BINARY)
	## -# Generate minor embedding
	# Embedding number of trials
	num_seeds = 10
	emb_seeds = np.random.randint(0, 100000, num_seeds)
	if Verbose:
		print(mysampler)
	# call the function for minor-embedding generation and selection
	embedding, numQubits = generate_embedding(model, sampler=mysampler, seeds=emb_seeds, return_num_qubits=True)
	return model, embedding, numQubits

## setLin2DwaveRegister set the DWave register to solve the Lin2Bench problem 
## - Inputs are:
## 	- sampler (DWave QPU ID)
##	- niter, iteration number 
##	- logfile (N.B.: not yet implemented TBA)
##	- verbose flag (default False)
## - Output: model (ready to run on QPU), associated minor-embedding, and number of qubits
def setLin2DwaveRegister(mysampler, niter, ctx, logfile=None, Verbose=False)-> BinaryQuadraticModel:
	## -# Prepare model by calling GnQUBO function from GnBenchLib
	n= niter
	myh, myJ, mycost= QUBOLin2Bench(n,ctx.resolution)
	h= getHforDimod(myh)
	J= getJforDimod(myJ)
	ctx.target= mycost[0] #< set the target to the computed optimal cost (solution of the linear system)
	model = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.SPIN)
	## -# Generate minor embedding
	# Embedding number of trials
	num_seeds = 10
	emb_seeds = np.random.randint(0, 100000, num_seeds)
	if Verbose:
		print(mysampler)
	# call the function for minor-embedding generation and selection
	embedding, numQubits = generate_embedding(model, sampler=mysampler, seeds=emb_seeds, return_num_qubits=True)
	return model, embedding, numQubits

# Additional imports for DWave QPU handling
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite

def runDwave(sampler= None, embedding= None, model= None):
	""" Use DWave procedures to solve the problem on the QPU 
	You can refer to DWave ocean documentation
	The principle is to use the "better" fixed minor-embedding found by the use of generate_embedding function
	input: valid sampler class refering to an actual DWave QPU, fixed embedding, and a QUBO/Ising model to run on the QPU 
	output: Dwave ocean sample class. Be carreful it is not directly compatible with the formatedSamples class from the BACQ libraries
	"""
	assert sampler != None
	assert embedding != None
	assert model != None
	embeddedSampler = FixedEmbeddingComposite(sampler, embedding)
	return embeddedSampler.sample(model, annealing_time=350, num_reads=1000)

def sampleSetDwave(myresponse=None):
	assert myresponse != None
	return myresponse

def formatSelectDwSample(dwsampleset, lowest= True):
	""" Transform DWave samples from Sample Ocean class to sampledResults class 
	input: DW sample 
	output: sampledResults samples
	"""
	minsamples= dwsampleset.lowest()
	mysample= sampledResults()
	mysample.info= minsamples.info
	#print("### ->", minsamples.record)
	mysample.energyCost= minsamples.record[0][1]
	print("lowest energy=", mysample.energyCost)
	samplerecord= minsamples.record
	sampletable= samplerecord[0]
	totsamples= 0
	lowsamples= 0
	#print(mysample.info)
	for cursample in dwsampleset.record:
		#print(cursample)
		totsamples+= cursample[2]
		if cursample[1] == mysample.energyCost:
			lowsamples+= cursample[2]
	mysample.totNum= totsamples
	mysample.num= lowsamples
	print("total sample num=", totsamples, lowsamples)
	#print(minsamples.variables, sampletable)
	mysample.numQubits= len(minsamples.variables)
	if mysample.numQubits != len(sampletable[0]):
		raise ValueError("Inconsistency in number of qubits and sample size in formatSelectDwSample")
	mysample.samples= np.asarray(sampletable[0])
	# check if only oneliner for returned sampleset
	if mysample.samples.ndim == 1:
		mysample.samples= mysample.samples.reshape(1,-1)
	# reorder columns to match the orders of variables
	order= np.argsort(minsamples.variables)
	mysample.samples= mysample.samples[:, order]
	print("this is my sample=", mysample, "\n", mysample.samples,"-> qbits=", mysample.numQubits)
	return mysample

	
import dimod
import argparse
import json

class DWaveBenchmark(QPUBenchClass):
	""" Concrete implementation of class QPUBenchClass for DWave family of QPUs 
	"""
	capabilitySet= ['qscore', 'gscore', 'lin2score', 'linsolverany', 'primescore'] ##< set of capabilities for DWave QPUs
	dwctx= QPUBenchContext() # initialize a QPU context 
	## initialize the remaining as default "uninitialized"
	dwsampler= None 
	dwembedding= None
	dwarch= None
	currentBench= None
	myPreprocess= None
	myCompute= None
	myPostprocess= None
	nIter=0
	preprocessRes= None
	computeRes= None
	isInit= False
	loginfile= None
	dat= None
	nqubits= -1
	dwaveDispatch= {'gscore':[setGnDwaveRegister, runDwave, evalGScore], 
		'lin2score':[setLin2DwaveRegister, runDwave, evalLin2Score], 
		'primescore':[preDwaveMultiplier, useDwaveMultiplier, returnDwavePrime]}
	
	@property
	def capabilities(self) -> QPUBenchClassCapability:
		## method to return capabilities of QPU benchmark implementations
		return QPUBenchClassCapability("DWave", capabilitySet)
	
	@classmethod
	def add_arguments(cls, parser: argparse.ArgumentParser):
		parser.add_argument('-l', '--logfile', default= 'dwaveBench.log', help='output log file (default= dwaveBench.log))')
		parser.add_argument('-a', '--arch', default= 'pegasus', help='Selection of architecture type of DWave QPU')
		parser.add_argument('-p', '--params', default= 'params.json', help='more parameters for the primescore')
	
	def is_available(self,qpuArch= None) -> bool:
		## Check if DWave libraries are available
		try:
			import dimod
			from dwave.system.samplers import DWaveSampler
			return True
		except ImportError:
			return False

	def selectBenchmark(self, bench: str, num:int, qpuArch= None, **kwargs):
		print("Select Benchmark=", bench, qpuArch, self.capabilitySet)
		self.dwctx.currentBench= bench
		# Example: select gscore at n=2
		print("type=", type(self.capabilitySet), type(bench))
		print("test=", bench in self.capabilitySet)
		print("test2= ", self.capabilitySet[1], bench, "-", len(bench), len(self.capabilitySet[1]))
		if bench in self.capabilitySet:
			self.currentBench= bench
			self.nIter= num
		else:
			print("Warning = None selected")
			return False
		if qpuArch != None:
			print("selecting sampler from arch=", qpuArch)
			self.dwsampler= selectDWaveQPU(qpuArch)
		processSet= self.dwaveDispatch[bench]
		self.myPreprocess= processSet[0]
		self.myCompute= processSet[1]
		self.myPostprocess= processSet[2]
		#print(processSet[2])
		return True

	def preprocessIteration(self, bench:str, **kwargs) -> bool:
		#from dwave.system.composites import EmbeddingComposite
		if self.myPreprocess == None:
			return False
		
		if self.currentBench == 'primescore' and self.params_file is not None:
			with open(self.params_file, 'r', encoding='utf-8') as f:
				self.dwctx.params = json.load(f)

		self.preprocessRes, self.dwembedding, self.nqubits= self.myPreprocess(self.dwsampler, self.nIter, self.dwctx)
		return True
		 

	def computeIteration(self, **kwargs) -> bool:
		# Example: gscore at n=2
		#print("this is compute=", self.myCompute)
		if self.myCompute == None:
			return False
		print("this is sampler=", self.dwsampler)
		self.computeRes= self.myCompute(self.dwsampler, self.dwembedding, self.preprocessRes)
		self.dwctx.computeRes= self.computeRes
		self.dwctx.formatedSamples= formatSelectDwSample(self.computeRes)
		return True

	def postprocessIteration(self, **kwargs) -> bool:
		if self.myPostprocess == None:
			print("what is happening? ", self.myPostprocess)
			self.postprocessRes= self.myCompute
		else:
			self.postprocessRes= self.myPostprocess(self.dwctx)
		return True
		
	def __init__(self, iteration, bench, arch, logfile=None, params=None, resolution=1):
		loginfile= logfile 
		dwarch= arch
		currentbench= bench
		self.dwctx.currentBench= bench
		self.dwctx.nIter= int(iteration)
		self.dwctx.arch= arch
		self.selectBenchmark( bench, int(iteration), dwarch)
		self.params_file = params
		self.resolution = resolution
		
	def preprocess(self):
		#self.dwsampler= 
		self.dwctx.nIter= self.nIter
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
		myret, myscore= self.myPostprocess(self.dwctx)
		return myret, myscore


####################

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

##print(logfile)
#n= myargs.start
#print(n)

#logf= open(logfile, "w", encoding="utf-8") # overwrite previous if it exists
## instantiation of QPUBenchClass using the collected data 

#myBench= DWaveBenchmark(1,tobench,setarch, tobench)

#benchdone= False

#while not benchdone:
#	print("############# ?????? nIter=", myBench.nIter)
#	myBench.preprocess()
#	myBench.compute()
#	benchdone, score= myBench.postprocess()
#	myBench.nIter+= 1	

#print("Final score on bench", myBench.currentBench, "is", score)

