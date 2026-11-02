## @package GnBenchLib
## @brief GnBenchLib is the generic (QPU independent) function sets for implementing the GScore.
## Dependencies are:
##	- numpy
##	- linear algebra for Ising module/library
##	- QPUBenchClass library
##

import numpy as np
import sys
# add project root to python path
from pathlib import Path
root = Path(__file__).resolve().parents[2]   
sys.path.insert(0, str(root))
# then we can import libraries from project
from lib.linear_algebra_ising import *
from lib.QPUBenchClass import *


#  Package to manage the Gn set aimed at the GScore Benchmark
# Below = Main set of functions to compute the Gn and exploit the results """

def maxValH(h):
	r""" Returns extremal value within a given vector
		- input: vector h
		- output: value from maximum absolute value found
	"""
	ret= 0
	for i in range(len(h)):
		val= abs(h[i])
		if val > ret:
			ret= val
	return ret

def GnQUBO(n, Verbose=False):
	r""" Generate Gn QUBO matrix
		 - Input: value of n (integer, positive) for the Gn 
		 - Output: Numpy QUBO matrix for the Gn using the value n provided as input """
	## -# First pass= numbering the nodes of the Gn graph
	
	# u_ji nodes have i and j as indices with j between 0 and n and same for i
	# (j is the column index)
	u=[[0 for j in range(0,n+1)] for i in range(0,n+1)]
	k=0
	for j in range(0,n+1):
	    for i in range(0,n+1):
	        u[j][i]=k
	        k=k+1
	        if Verbose:
	        	print('u(',j,',',i,')=>',u[j][i])
	if Verbose:
		print()
	
	# v_ji nodes have also j and i as indices
	# (as in the previous case, j is the column index and i the row index)
	v=[[0 for j in range(0,n+1)] for i in range(0,n+1)]
	for j in range(0,n+1):
	    for i in range(0,n+1):
	        v[j][i]=k
	        k=k+1
	        if Verbose:
	        	print('v(',j,',',i,')=>',v[j][i])
	
	## -# Second pass= on edges between u and v nodes

	# 1/ u[j][i] has a connection with v[j][i] for all j from 0 to n and i from 0 to n.
	E=[]
	for j in range(0,n+1):
	    for i in range(0,n+1):
	        E.append({u[j][i],v[j][i]})

	# 2/ v[j][i] has a connection to u[j+1][k] for all j from 0 to n-1, i from 0 to n and k from 0 to 	n        
	# these are the "trap"/dense connections in the Gn graph
	for j in range(0,n):
	    for i in range(0,n+1):
	        for k in range(0,n+1):
	            E.append({v[j][i],u[j+1][k]})

	## -# Finally= generate QUBO matrix from graph
	N= len(E)
	L= N
	numval=0
	Q=[[0 for e1 in range(0,N)] for e2 in range(0,N)]

	for e1 in range(0,N):
    		for e2 in range(e1,N):
        		if e1==e2:
            			Q[e1][e2]=1+2*L # value on the diagonal
        		elif E[e1].intersection(E[e2])!=set():
                			Q[e1][e2]=-2*L  # non 0 values off-diagonal
                			numval+=1

	myQ= np.array(Q)*(-1)
	return myQ, numval

def edgestosummitGn(n):
	r"""
	Compute the corresponding sets between Gn nodes and Gn edges
		Input: value of n for Gn 
		Output: double vector for corresponding set between edges and nodes
	"""	
	e2s= np.zeros(((n+1)**3,2),int)
	# first (n+1)^2 edges (part of best solution)
	edgnum= 0
	for i in range((n+1)**2): # optimal solution
		e2s[edgnum][0]= i
		e2s[edgnum][1]= i+(n+1)**2
		edgnum+= 1
	# second: non optimal solution edges 
	for i in range(n): # loop on the dense parts, the number is n
		for j in range(n+1): # loop on the v nodes of each dense part
			for k in range(n+1): # loop on the v nodes of each dense part
				#print(edgnum, i, j, k, k+(i+1)*(n+1), (n+1)**2+j+i*(n+1))
				e2s[edgnum][0]= k+(i+1)*(n+1)
				e2s[edgnum][1]= (n+1)**2+j+i*(n+1)
				edgnum+= 1
	return e2s
	

## generate simplified QUBO matrix with values in {-1,0,1}
## input: numpy matrix for QUBO
## output: simplified QUBO matrix (numpy)
def simpleQUBO(E):
	R= np.zeros((len(E),len(E)))
	for i in range(len(E)):
		for j in range(len(E[i])):
			R[i,j]= np.sign(E[i,j])
	return R

## define the adjacency matrix for MIS solving	
## input: n value for Gn 
## output: adjacency matrix for the MIS formulation of Gn 
def GnMISadjacency(n):
	myQ, numval= GnQUBO(n) 
	size= len(myQ)
	myMIS= np.zeros((size,size))
	for indi in range(size):
		for indj in range(size):
			if (myQ[indi,indj] != 0) and (indi != indj):
				myMIS[indi,indj]= 1
				myMIS[indj,indi]= 1
	return myMIS

# Set of auxillary functions to compute the GScore defined below 

def hammingDist(n, sample):
	r""" Compute the Haming distance from optimal solution of Gn series in standard numbering of edges to the actual found solution
	Input: value of n for Gn , vector candidate solution
	Output: integer= Hamming distance= number of discrepencies in the bit string of the provided vector """
	d= 0
	inc= 0
	tresh= (n+1)*(n+1) # in the optimal solution all the (n+1)^2 first variables are set to 1 and all the others set to 0
	for i in sample:
		if inc < tresh: # solution 1 for the optimal (below [n+1]^2)
			if i != 1:
				d+=1 # discrepency found
		else:           # solution 0 for the optimal (from [n+1]^2)
			if i != 0: 
				d+=1 # discrepency found
		inc+= 1
	return d
	
def lowestHammingDist(n, sample):
	r""" Search the set of samples with the lowest Haming distance to the optimal
	- input 1: n value of G_n 
	- input 2: sample or set of samples 
	- output: lowest Haming distance in the sample/sample set.
	"""
	print(type(sample))
	if sample.ndim == 2:
		dh= len(sample[0])+1
		#print(">> case 2 in lowestHammingDist")
		for s in sample:
			d= hammingDist(n,s)
			if d<dh:
				dh=d
	elif sample.ndim == 1:
		#print(">> Case 1 in lowestHammingDist")
		dh= hammingDist(n,sample)
	else:
		raise(f"Badly formatted samples in lowestHammingDist {sample}")
	return dh
	

def brokenConstraints(n, sample):
	r""" 
	Compute the number of broken constraints in the candidate vector solution
	- Input: integer value of n from Gn, binary vector candidate solution
	- Ouput: integer, number of broken constraints
	- prerequisite: the numbering of nodes is the traditional one; the (n+1)**2 first variables are the optimal solution 
	""" 
	br= 0
	node= 0
	mymap= edgestosummitGn(n)
	mylinks= np.zeros(2*(n+1)**2, int)
	# count the number of selected edges per node
	for i in sample:
		if i == 1:
			nu= mymap[node][0]
			nv= mymap[node][1]
			mylinks[nu]+= 1
			mylinks[nv]+= 1
		node+= 1
	# count the number of nodes with more than 1 edge
	for edgenum in mylinks:
		if edgenum > 1:
			br+= 1
	# return the number of broken constraints
	return br

def lowestBrokenConstraints(n, sample):
	r""" Search the lowest set of samples with the lowest number of broken constraints
	- input 1: n value of G_n 
	- input 2: sample set of samples 
	- output: lowest number of broken constraints in the sample set.
	"""
	#print(type(sample))
	if sample.ndim == 2:
		bc= len(sample[0])+1
		#print(">> case 2 in lowestBrokenConstraints")
		for s in sample:
			b= brokenConstraints(n,s)
			if b<bc:
				bc=b
	elif sample.ndim == 1:
		#print(">> Case 1 in lowestBrokenConstraints")
		bc= brokenConstraints(n,sample)
	else:
		raise(f"Badly formatted samples in lowestBrokenConstraints {sample}")
	return bc
	
def gnScore(nfailed, proba, sample, redux=0.0):
	r""" Compute GScore from 
	- input 1: the failed n rank (nfailed), 
	- input 2: the probability of finding the correct answer at nfailed-1
	- input 3: the best set of samples
	- input 4 (optional): for approximate embedding, the reduction score to be applied (default 0 = none)
	- output: returns the evaluation of the G-Score
	"""
	print("GnBenchLib gnScore: proba=", proba)
	# Hamming distance and subsequent Hamming ratio
	hd= lowestHammingDist(nfailed, sample.samples[0])
	rh= np.float64(hd)/(nfailed+1.0)**3.0
	# standard case: use Number of failed constraints and constraint success ratio
	if redux == 0.0: 	
		nb= lowestBrokenConstraints(nfailed,sample.samples[0])
		if nb > (nfailed+1)*(nfailed+1):
			rc= 1
		else:
			rc= nb/((nfailed+1.0)*(nfailed+1.0))
		ro= (2-rc-rh)*0.5
		gs= (nfailed+ro)**3 # was (nfailed+proba*ro)**3
	else: # in the case we are on approximate embedding, only use hamming distance and redux ratio
		ro= (1-rh)
		gs= (nfailed+ro)**3-redux
		rc= 1
	print("GnBenchLib gnScore=", nfailed, ro, rc, gs)
	return gs

def successProbaGn(n, samples: sampledResults):
	"""
	Return an estimator of the success probability for the Gn optimization outcome
	- Inputs : n value of Gn, and samples
	- Output: estimated probability (takes into account insufficient statistical relevance)
	"""
	num= samples.totNum
	mincost= samples.energyCost
	smpl= samples.samples[0]
	print("# GnBenchLib-successProbaGn",n, smpl)
	size= len(smpl)
	nummin= samples.num
	dh= lowestHammingDist(n, smpl)
	print("#@ GnBenchLib-successProbaGn", dh, nummin, num)
	if dh== 0:
		successprob= np.float64(nummin)/np.float64(num)
		if successprob < 0.2/(n+1)**3: # too close to random
			successprob= 0
		elif nummin < 3: # not enough statistic, maybe lucky
			successprob= 0
	else:
		successprob= 0
	return successprob

	
def lowestCost(n, samples):
	"""
	Search minimal cost from samples for Gn 
	- input: n value for Gn, and set of total samples
	- return: set of samples with the minimal QUBO cost
	"""
	mincost= 1e6
	ind= 0
	for smpl in samples.record:
		if smpl[1] < mincost:
			minind= ind
			mincost= smpl[1]
		ind+= 1
	# now agregate minimum costs
	mysmpl= {}
	for smpl in samples.record:
		if smpl[1]== mincost:
			mysmpl+= smpl
	return mysmpl

"""
 @evalGScore is the base for computing the GScore in an incremental fashion.
 If all goes well, it is possible to set it as the postprocessing method
 (QPU postprocessing once the samples are selected and formatted)
"""
def evalGScore(ctx):
	print("# GnBenchLib evalGScore n=",ctx.nIter)
	ncurrent= ctx.nIter
	prevproba= ctx.dat 
	resSample= ctx.computeRes
	print("#@ GnBenchLib evalGScore context=", ncurrent, prevproba)
	# sanity checks
	if ctx.currentBench != 'gscore':
		# nice try but no
		raise ValueError(f"Call of evalGScore for currentBench {ctx.currentBench} instead of 'gscore'") 
	if len(resSample) == 0 or ncurrent < 1 or prevproba < 0 or prevproba > 1:
		# Big issue= no provided sample, incorrect utilization of function
		raise ValueError("Incorrect argument call in evalGScore") 
	# Default return value is False (= end of evaluation not reached yet)
	retF= False
	# retrieve lowest energy samples from context
	mysample= ctx.formatedSamples
	# update probability
	theSuccessProb= successProbaGn(ncurrent, mysample)
	print("#@ GnBenchLib evalGScore success probability=",theSuccessProb*100, "#\n")
	# update GScore and return success or not at rank ncurrent
	if theSuccessProb == 0:
		theSuccessProb= prevproba
		retF= True
	else:
		ctx.dat= theSuccessProb
	# compute GScore
	if retF:
		# final increment, failed best configuration
		retg= gnScore(ncurrent, theSuccessProb, mysample)
	else:
		# still going, the score is at least this
		retg= (ncurrent+1)**3
	return retF, retg


def simplification_score(A: np.ndarray,
                         A_prime: np.ndarray,
                         lambda_pen: float = 1) -> (float, int):
    r"""
    Evaluate graph simplification in terms of equivalent number of nodes.
    
    The returned score lies in [alpha, n], where:
      - n     = number of nodes in the original graph
      - alpha = cardinality of a maximum independent set (or a lower bound)
    
    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of the original graph (n x n)
    A_prime : np.ndarray
        Adjacency matrix of the transformed graph
        (same size with zero rows for pruned nodes, or smaller)
    lambda_pen : float
        Penalty factor applied to degree increases (default = 1. as maybe for future extension)
    
    Returns
    -------
    tuple of boolean and float:
        - float= score reductio ratio in the simplified graph wrt the initial graph
        - int= total number of edges in the intial graph = boolean
    """
    
    A = np.asarray(A, dtype=float)
    A_prime = np.asarray(A_prime, dtype=float)
    
    N = A.shape[0]
    
    # Handle size mismatch (pruning by removing rows/columns)
    # N.B: this should not happen (be sure to match the numbering and the size of the initial matrix)
    if A_prime.shape[0] != N:
        A_full = np.zeros((N, N), dtype=float) # 0 padding (/!\ can change your initial qubit labels)
        k = A_prime.shape[0]
        A_full[:k, :k] = A_prime
        A_prime = A_full
    
    # Degrees per row 
    deg_G  = A.sum(axis=1)
    deg_Gp = A_prime.sum(axis=1)
    
    diff=0
    # and sum the differences *in absolute values*
    for g,gp in zip(deg_G,deg_Gp):
    	if g >= gp:
    		diff+= g-gp
    	else:
    		# Penalized complexity 
    		# (the lambda factor for augmented penalties is a possible extension, not yet used in production)
    		diff+= (gp-g)*lambda_pen
    # in case there is no discrepencies, return OK for embedding and 0 penality
    if diff == 0: 
        return 0
    
    print(">> <<", diff)
    # Total number of nodes for the original graph 
    dtot= deg_G.sum()
    # Evaluation of the ratio of edge reduction after simplification
    scoreredux = diff / dtot
    
    return (scoreredux, # returns calculated reduction ratio to the gscore due to graph simplification (float)
        dtot)	# also returns the number of edges in the original graph (integer)

def scoreRedux(n: int, A: np.ndarray, Ap: np.ndarray)-> float:
	# sanity check 
	N = A.shape[0]
	assert N == (n+1)**3
	
	reduxRatio, dtot= simplification_score(A,Ap)
	print(">>",dtot, n*(n+1)*(n+1)*(n+2)*2)
	assert dtot == n*(n+1)*(n+1)*(n+2)*2 # second sanity check
	print(reduxRatio)
	sredux= reduxRatio*N
	
	return sredux
	

"""
 Main program: unitary and integration tests (when not used as a library)
"""

if __name__ == "__main__":
	import sys 
	from lib.linear_algebra_ising import *
	if len(sys.argv)<2:
		print(sys.argv," n\n with n the number n in the Gn series");
		sys.exit()

	n=int(sys.argv[1])
	print("Computing values of G",n)

	print("---")
	print(edgestosummitGn(n))
	print("---")
	myQ,numval= GnQUBO(n)
	N= len(myQ)
	h,J= QUBOtoIsing(myQ)
	
	print('Matrice QUBO :')                
	for e1 in range(0,N):
	    for e2 in range(0,N):
	        print(myQ[e1][e2],end=' ')
	    print()
	
	simpQ= simpleQUBO(myQ)
	simpIsingh, simpIsingJ= QUBOtoIsing(simpQ*4)
	print(simpQ)

	print("\n------QUBO---------\n")
	QUBOoutput(simpQ)
	print("\n------Ising---------\n")
	IsingOutput(simpIsingh, simpIsingJ)
	print("\n---------------\n")
	max= maxValH(h)
	for i in range(len(h)):
		print(2*h[i]/max,", ",end=' ')
	print("***** ")
	for i in range(len(J)):
		val= 0
		for j in range(len(J)):
			if (J[i][j] != 0) & (J[i][j]!=val):
				print(f"({i},{j}):",2*J[i][j]/max,end=', ')
				val= J[i][j]
		print(" ")
	print("-----------")
	print("val16=",numval)

	print("\nhamiltonian= SparsePauliOp.from_list([",end='')
	myl= len(h)
	tbc= False
	# first: diagonal elements with vector h
	for i in range(myl):
		if h[i] !=0 :
			# check if a comma is needed
			if tbc:
				print(",",end='')
			print("(\"",end='')
			for k in range(len(h)):
				if (myl-i-1) ==k:
					print("Z",end='')
				else:
					print("I",end='')
			print(f"\",{str(6*h[i]/max)[1:-1]})",end='')
			tbc= True
	# second: non diagonal elements, matrix J
	for i in range(myl):
		for j in range(myl):
			if J[i][j] != 0:
				# check if a comma is needed
				if tbc:
					print(",",end='')
				print("(\"",end='')
				for k in range(myl):
					if (myl-1-i) !=k and (myl-1-j) !=k:
						print("I",end='')
					else:
						print("Z",end='')
				print(f"\",{str(6*J[i][j]/max)[1:-1]})",end='')
				tbc= True
	print("])")
	print("-----MIS adjacency matrix")
	A= GnMISadjacency(n)
	print(A)
	if n== 2:
		A2_3d= np.array([
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #0 		#0
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #1 		#0
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #2		#0
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0 ], #3		#0
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0 ], #4		#2
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0 ], #5		#2
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1 ], #6		#0
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0 ], #7		#0
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1 ], #8		#0
		[ 1, 1, 1, 0, 0, 0, 0, 0, 0,  0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #9		#1
		[ 1, 0, 0, 0, 0, 0, 0, 0, 0,  1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #10		#0
		[ 1, 1, 0, 0, 0, 0, 0, 0, 0,  0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #11		#2
		[ 0, 0, 0, 1, 0, 0, 0, 0, 0,  1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #12		#0
		[ 0, 1, 0, 1, 1, 0, 0, 0, 0,  0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #13		#1
		[ 0, 1, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #14		#0
		[ 0, 0, 1, 1, 0, 1, 0, 0, 0,  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #15		#1
		[ 0, 0, 1, 0, 0, 0, 0, 0, 0,  1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #16		#0
		[ 1, 0, 1, 0, 0, 0, 0, 0, 0,  0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #17		#2
		[ 0, 0, 0, 1, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0 ], #18		#0
		[ 0, 0, 0, 0, 0, 0, 1, 1, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0 ], #19		#1
		[ 0, 0, 0, 1, 0, 1, 0, 0, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0 ], #20		#1
		[ 0, 0, 0, 0, 0, 0, 1, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 ], #21		#0
		[ 0, 0, 0, 1, 1, 0, 0, 1, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ], #22		#1
		[ 0, 0, 0, 0, 0, 0, 0, 0, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1 ], #23		#0
		[ 0, 0, 0, 0, 0, 0, 1, 1, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0 ], #24		#2
		[ 0, 0, 0, 0, 0, 0, 0, 1, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0 ], #25		#0
		[ 0, 0, 0, 0, 0, 0, 1, 0, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0 ], #26		#2
		])
		A2_2d= np.array([
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #0 		#0
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #1 		#0
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #2		#0
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0 ], #3		#0
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ], #4		#4
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ], #5		#4
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1 ], #6		#0
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0 ], #7		#0
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1 ], #8		#0
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #9		#6
		[ 1, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #10		#1
		[ 1, 1, 0, 0, 0, 0, 0, 0, 0,  0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #11		#2
		[ 0, 0, 0, 1, 0, 0, 0, 0, 0,  0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #12		#1
		[ 0, 1, 0, 1, 1, 0, 0, 0, 0,  0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #13		#1
		[ 0, 1, 0, 0, 0, 0, 0, 0, 0,  0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #14		#1
		[ 0, 0, 1, 1, 0, 1, 0, 0, 0,  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #15		#1
		[ 0, 0, 1, 0, 0, 0, 0, 0, 0,  0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #16		#1
		[ 1, 0, 1, 0, 0, 0, 0, 0, 0,  0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #17		#2
		[ 0, 0, 0, 1, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0 ], #18		#1
		[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], #19		#6
		[ 0, 0, 0, 1, 0, 1, 0, 0, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0 ], #20		#1
		[ 0, 0, 0, 0, 0, 0, 1, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 ], #21		#1
		[ 0, 0, 0, 1, 1, 0, 0, 1, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ], #22		#1
		[ 0, 0, 0, 0, 0, 0, 0, 0, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1 ], #23		#1
		[ 0, 0, 0, 0, 0, 0, 1, 1, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0 ], #24		#2
		[ 0, 0, 0, 0, 0, 0, 0, 1, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0 ], #25		#1
		[ 0, 0, 0, 0, 0, 0, 1, 0, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0 ], #26		#2
		])
		redux2= scoreRedux(2, A, A2_2d)
		pseudosample= sampledResults(np.array([[1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]),1,1,1E6,27,None)
		redux3= scoreRedux(2, A, A2_3d)
		print("Result best case for 2D pasqal", 27-redux2, "gscore max=", gnScore(2,1.0, pseudosample, redux2))
		print("Result best case for 3D pasqal", 27-redux3, "gscore max 3D=", gnScore(2, 1.0, pseudosample, redux3))
