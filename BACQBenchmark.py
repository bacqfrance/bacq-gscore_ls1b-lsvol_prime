"""
Main stub for running BACQ benchmarking suite.
Allows multiple benchmarks and multiple backends (QPUs)
"""

import importlib

backendMap= {"dwave": ("DWaveBenchmark", "DWaveBenchmark"), 
	"cpu": ("CPUBenchmark", "CPUBenchmark")
	}

## BACKENDS is a list of available backends
BACKENDS= {}

import hardware.dwave.DWaveBenchmark

for backendName, (moduleName, className) in backendMap.items():
	try:
		## try and import relevant backend library
		modpath= f"hardware.{backendName}.{moduleName}"
		#print(modpath)
		module= importlib.import_module(modpath, package=__name__)
		## get class within module 
		backendClass= getattr(module, className)
		## update BACKENDS dictionnary 
		BACKENDS[backendName]= backendClass
	except  Exception as e: 
		## ignore if not available or error at loadingtime
		print(f"Warning: backend {backendName} not available")
		print(f"Error: {type(e).__name__}: {e}")
		continue
		

import argparse
from pathlib import Path
import os
import sys
import json

## Main implementation of BACQ Benchmark 		
## - Stub for every possible backend
def main():
	## - Parser main (global and common) options
	parser= argparse.ArgumentParser(description= "BACQ benchmark with several QPU backends")
	parser.add_argument('-v', '--verbose', action='store_true', help="Verbose mode")
	parser.add_argument('-i', '--iteration', default= 1, help="Iteration number of the selected benchmark")
	parser.add_argument('-b', '--bench', default='gscore', help="Select benchmark from benchmark suite [default= gscore]")
	parser.add_argument('-r', '--resolution',type=int,default=1,help='Resolution in bits for Lin2Bench'
)
	## - create subparsers for possible backends
	subparsers = parser.add_subparsers(dest='backend_choice', required=True, help="Select backend among these choices")
	
	## - option parsing depending on the requested backend
	for name, backendClass in BACKENDS.items():
		subparser= subparsers.add_parser(name, help=f"Options for bakckend {name}")
		##     - then the bakckend adds its own arguments
		backendClass.add_arguments(subparser)
	## - parsing arguments from CLI
	args= parser.parse_args()
	

	## - Dynamic instantiation of the selected backendClass
	##     - For that use the class corresponding to the user's choice
	backendClass= BACKENDS[args.backend_choice]
	
	## - Remove general configuration flags to send only backend specific options to selected backend
	backendVars= vars(args).copy()
	backendVars.pop('backend_choice')
	backendVars.pop('verbose')
	
	## - Magic instantiation through CLI arguments
	backendBench= backendClass(**backendVars)
	backendBench.dwctx.resolution = args.resolution
	## - Backend execution (generic and use specific implementation through class instantiation)
	#backendBench.run()
	benchdone= 0

	while not benchdone:
		print("****** current nIter=", backendBench.nIter)
		backendBench.preprocess()
		backendBench.compute()
		benchdone, score= backendBench.postprocess()
		if benchdone == -1:
			benchdone=0
		elif benchdone == 0:
			backendBench.nIter+= 1	

	print("Final score on bench", backendBench.currentBench, "is", score)


if __name__ == "__main__":
	main()
