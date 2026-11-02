#!/bin/python3
 
## @file QPUBenchClass.py
## @package QPUBenchClass
## @brief QPUBenchClass is the set of reference classes for project BACQ refering to management of multiple benchmark scores and multiple targets.
## Dependencies are:
##	- ABC
##	- typing
##	- numpy
##	- argparse
##	- dataclass
##
## Distributed under EU Public Licence (EUPL) and (c) CEA LIST, QuLIST team

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, List
import inspect
from dataclasses import dataclass
import argparse
import numpy as np

@dataclass
class QPUBenchClassCapability:
	name: str
	supports: List[str]  # e.g., ["qscore", "gscore", "linsolver"]
	
@dataclass
class QPUBenchContext:
	nIter= -1 	##< Iteration number on the current benchmark score
	dat= 0	##< secondary value relevant for the benchmark, versatile use whose signification depends on the benchmark
	target= 100000 ##< target value (optional)
	proba= -1	##< evaluated probability of outcome for the given computing
	computeRes= None	##< bare bone results from QPU computing (before formatting)
	currentBench= None	##< name of the current benchmark (e.g. 'gscore')
	arch= None		##< architecture name
	subarch= None	##< subarchitecture name (within architecture what specific QPU is used)
	formatedSamples= None	##< current samples (outcome for GPU, but formatted and selected)
	probaDist= None 	##< list of probability for statistical analysis (depending on benchmark program)
	distVectors= [[],[],[]]


@dataclass
class samplesetClass:
	vecVal: np.ndarray
	energy: float
	occurences: int= 1

@dataclass
class sampledResults:
	samples: samplesetClass= None
	num: int= -1
	totNum: int= -1
	energyCost: float= 1E6
	numQubits: int= -1
	info: Any= None
	
class QPUBenchClass(ABC):
	@property
	@abstractmethod
	def capabilities(self) -> QPUBenchClassCapability:
		pass
        
	@abstractmethod
	def selectBenchmark(self, bench: str, **kwargs):
		pass

	@abstractmethod
	def preprocess(self, data: Any, **kwargs) -> Any:
		pass

	@abstractmethod
	def compute(self, data: Any, **kwargs) -> Any:
		pass

	@abstractmethod
	def postprocess(self, data: Any, **kwargs) -> Any:
		pass

	@abstractmethod
	def is_available(self) -> bool:
		r"""Check if the QPU is available at runtime."""
		pass        

	@classmethod
	def add_arguments(cls, parser:argparse.ArgumentParser):
		r"""Generic class method to add arguments to the cli invocation depending on instancied class. """
		pass

"""
 Main program: integration test (when not used as a library)
"""
if __name__ == "__main__":
	print("Base class reference for BACQ QPU benchmarking\n\t* Works as intended if this message is displayed without error")
	import sys
	sys.exit(True)

