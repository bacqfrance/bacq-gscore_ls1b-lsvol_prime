#!/bin/python3
 
## @file LinBench.py
## @package LinBench2
## @brief LinBench2 is the implementation of function sets for implementing Lin2Bench and LinNBench benchmarks.
## Dependencies are:
##	- numpy
##	- linear algebra for Ising module/library
##	- QPUBenchClass library
##
## Distributed under EU Public Licence (EUPL) 1.2 and (c) CEA LIST, QuLIST team

import numpy as np
from scipy.optimize import curve_fit
import sys
import random
# add project root to python path
from pathlib import Path
root = Path(__file__).resolve().parents[2]   
sys.path.insert(0, str(root))
# then we can import libraries from project
from lib.QPUBenchClass import sampledResults, QPUBenchContext
from lib.linear_algebra_ising import *


## Generate matrix A and vector B for Ax+B=0 equation from the planned solution Xmat
## - input Xmat: numpy vector target solution of the equation (in the benchmark we start from the solution)
## - input dyn: integer is the range of random values for the 
def generateAB(Xmat, dyn=5, sign=1):
	myr= len(Xmat)
	Amat= np.zeros((myr, myr))
	Bmat= np.zeros((myr,1))
	det=0
	while det == 0:
		for i in range(myr):
			val=0
			for j in range(myr):
				rd= int(random.uniform(-dyn-1,dyn+1))
				Amat[i,j]= rd
				val+= -Amat[i,j]*Xmat[j,0]
			Bmat[i,0]= val
		det= np.linalg.det(Amat)
	return Amat,Bmat


## Find 2's complement solution 
def find2bitSolutions(Qmat):
	myr= len(Qmat)
	Xvect= np.zeros((myr,1))
	mincost= sys.maxsize
	Xsol= np.zeros(myr)
	for varval in range(1<< myr):
		cost= 0
		for varind in range(myr):
			Xvect[varind]= ((varval & (1<<varind)) != 0)
			#print("    ",varval,",",varind," = ", 1<<varind, " ", Xvect[varind])
		for i in range(myr):
			for j in range(myr):
				cost+= (Qmat[i,j])*Xvect[i]*Xvect[j]
		print(varval,"-> ",cost," ",Xvect.transpose())
		if (cost<mincost):
			mincost= cost
			Xsol= Xvect.copy()
	return mincost, Xsol
	
## Generate binary matrices (2's complement encoding) from matrix A and vector B
def truncate2bits(Amat,Bmat):
	myr= len(Bmat)
	Cmat= np.zeros((myr,myr))
	Dmat= np.zeros((myr,1))
	for i in range(myr):
		for j in range(myr):
			Cmat.itemset((i,j),int(Amat[i,j])&1)
		Dmat.itemset((i,0),int(Bmat[i])&1)
	return Cmat,Dmat

## Generate a random solution vector of size myr and range in dyn
## inputs: myr (int, size of vector solution), dyn (int, range for random number generation)
def generateXrandom(myr, dyn=20):
	myX= np.zeros((myr,1))
	for i in range(myr):
		myX[i]= int(random.uniform(0,dyn+1))
	return myX

## Gives the minimum cost (optimal target) for the QUBO problem associated with AX+C=0
## - input: solution vector vecX (numpy vector), number of bits (resolution, int), h vector (numpy vector) and J (numpy matrix)	
def valcost(vecX, numbit, h, J):
	#print("valcost:",h.transpose(), J)
	rankX= len(vecX)
	numbX= numbit*rankX
	myx= np.zeros((numbX,1))
	cost= 0
	for i in range(rankX):
		for j in range(numbit):
			ind= i*numbit+j
			myx[ind,0]= (int(vecX[i,0])&(1<<j) !=0)*2-1
			cost+= myx[ind]*h[ind]
	#print("myX= ",myx.transpose())
	for i in range(numbX):
		for j in range(numbX):
			cost+= J[i][j]*myx[i]*myx[j]
	return cost

## LinProb is the frontend function to implement the Lin*Bench benchmark 
## LinProb provides a random X vector solution and the problem instance that
## corresponds A matrix and B vector so that AX+B=0
## - myrange is the current number of variables in the iterations
## - bitnum is the current target resolution of the unknowns in number of bits
## - returns A matrix, and B and X vectors
def LinProb(myrange, bitnum):
	## -# Step 1: Compute bit mask for the solution vector X so that the solution is below \f$2^{n-1}\f$
	maxr= (1<<bitnum)-1
	## -# Step 2: generate random solution with the given bitmask
	myX= generateXrandom(myrange+2, maxr)
	print("LinProb: X=", myX.transpose())
	## -# Step 3: generate A matrix and B vector so that AX+B=0 
	myA,myB= generateAB(myX)
	#print("LinProb: B=", myB.transpose(), "\nA=", myA)
	## -# Final: returns the A matrix, B vector and X vector solution
	return myA, myB, myX 
	
## Lin2Prob is the frontend function to implement the Lin2Bench benchmark 
## It provides the A matrix, B vector and X solution vector with \f$X\in \{0,1\}^n\f$ 
## - myrange is the current number of variables in the iterations.
## - returns A matrix, and B and X vectors
def Lin2Prob(myrange,r=1):
	return LinProb(myrange, r)

## QUBOLin2Bench is one of the backend function to implement the Lin2Bench benchmark 
## It gives a QUBO formulation of an instance of the problem to solve
## - myrange is the current number of variables in the iterations.
## - returns QUBO matrix in h & J form, and associated minimal cost value 
def QUBOLin2Bench(myrange,r=1):
	## - First: get the problem instance
	myA, myB, myX= Lin2Prob(myrange,r) 
	
	## - Second: compute QUBO formulation and minimum cost of the cost function
	Qmat= QUBOfy(myA,myB, r) 
	#myh= getHQUBOforDimod(Qmat)
	#myJ= getJforDimod(Qmat)
	myh,myJ= QUBOtoIsing(Qmat)
	mycost= valcost(myX, r, myh, myJ)
	print("Min cost= ", mycost)
	#raise("stop here for now")
	## - return J matrix, h vector (dimod style) and calculated cost
	return myh, myJ, mycost

## sucessProbaLin2 implements the evaluation of the success probability on one given instance of a problem
## it will rely on the already available optimal score computed in the previous function (QUBOLin2Bench)
## - input: ctx (current context) and sample (the current set of selected samples)
## - output: evaluated probability (note that there will be a need for statistical analysis of the results)
def successProbaLin2(ctx, mysample):
	print(f"succesProbaLin2: {ctx.target} vs {mysample.energyCost}\n\t {mysample.num}\t {mysample.totNum} and {mysample.samples}")
	retproba= 0.0
	if ctx.target == mysample.energyCost:
		retproba= float(mysample.num)/float(mysample.totNum)
	return retproba
	
## fitSigmoid= fit a sigmoide function with the set of values \f$f(x)=\frac{1}{1+e^{-\lambda(x-x_0)}}\f$
## At the end the relevant value will be \f$x_0\f$ with its evaluated standard deviation
## But we also return the \lambda value with its standard deviation
## - input: QPU context with the distribution
## - optional input: file name to plot a graph (defaut None and no output graph)
## - output: returns estimated lambda with its standard deviation and x_0 with its standard deviation
def fitSigmoid(ctx, filename= None):
	## -# Get experimental results from context
	x = np.array(ctx.distVectors[0])+2
	y = np.array(ctx.distVectors[1])
	sigma_y = np.array(ctx.distVectors[2])
	assert len(x)== len(y) and len(x)==len(sigma_y)
	
	## -# Use sigmoid reciprocal function to fit a line, usign a change of variable: Y = ln(1/y - 1)
	Y = np.log((1.0 - y) / y)

	## -# Propagate evaluated error on y according to the change of variable (limited development) : sigma_Y = sigma_y / (y * (1 - y))
	sigma_Y = sigma_y / (y * (1.0 - y))
	
	## -# define a linear model to fit Y = a*x + b
	def linearModel(x, a, b):
	    return a * x + b
	
	## -# Linear regression with propagated uncertainties
	popt, pcov = curve_fit(linearModel, x, Y, sigma=sigma_Y, absolute_sigma=True)
	
	a, b = popt
	sigma_a, sigma_b = np.sqrt(np.diag(pcov))
	
	## -# Extract parameters of the sigmoid (\lambda and x0)
	lambda_est = a
	x0_est = -b / a

	print(f"(lambda) value = {lambda_est:.4f} ± {sigma_a:.4f}")
	print(f"x0 evaluated         = {x0_est:.4f}")

	## -# Set covariance matrix values to calculate standard deviation on x0
	sigma_a2 = pcov[0, 0]        # Variance of a
	sigma_b2 = pcov[1, 1]        # Variance of b
	cov_ab   = pcov[0, 1]        # Covariance between a and b

	## -# Calculate variance of x0 
	var_x0 = (1.0 / a**2) * ( (x0_est**2) * sigma_a2 + sigma_b2 + 2 * x0_est * cov_ab )
	sigma_x0 = np.sqrt(var_x0)

	print(f"x0 = {x0_est:.4f} ± {sigma_x0:.4f}")
	if filename != None:
		# compute fitted sigmoid
		import matplotlib.pyplot as plt
		fitx= []
		fity= []
		for myx in range(x[0],x[-1]):
			fitx.append(myx)
			fity.append(1/(1+np.exp(lambda_est*(myx-x0_est)))) 
		sigx= np.array(fitx)
		sigy= np.array(fity)
		# set figure
		plt.figure(figsize=(8,5))
		plt.errorbar(x,y,sigma_y, fmt='-o', color='C0', 
			ecolor='gray', elinewidth=1.5, capsize=4, markersize=7, label= 'QPU outcome', zorder=2)
		# set fit 
		plt.plot(sigx, sigy, color='red', linewidth=2, label='Sigmoid Fit', zorder=1)
		plt.xlabel('Number of variables')
		plt.ylabel('Probability of success')
		plt.title('QPU results Lin2Bench and sigmoid fit')
		plt.grid(True, alpha= 0.3)
		plt.legend()
		# save graph
		plt.savefig(filename, format='svg', bbox_inches='tight')
		plt.show()
	return lambda_est, sigma_a, x0_est, sigma_x0

## evalLin2Score function implements the Lin2Bench score evaluation in an incremental fashion
## To be set as post-processing method according to QPU call and management
## - input: ctx is the QPU general context after computation
def evalLin2Score(ctx):
	# Init
	print("# LinBenchLib evalLin2Score n=",ctx.nIter)
	ncurrent= ctx.nIter
	itcurrent= ctx.dat 
	resSample= ctx.computeRes

	print("#@ LinBenchLib evalLin2Score context=", ncurrent, itcurrent)
	# sanity checks
	if ctx.currentBench != 'lin2score':
		# nice try but no
		raise ValueError(f"Call of evalLin2Score for currentBench {ctx.currentBench} instead of 'lin2score'") 
	if len(resSample) == 0 or ncurrent < 1 or itcurrent < 0 or itcurrent > 1000:
		# Big issue= no provided sample, incorrect utilization of function
		raise ValueError("Incorrect argument call in evalLin2Score") 
	# Default return value is False (= end of evaluation not reached yet)
	retF= False
	
	## - retrieve lowest energy samples from context
	mysample= ctx.formatedSamples

	## - update probability
	theSuccessProb= successProbaLin2(ctx, mysample)
	print("#@ GnBenchLib evalGScore success probability=",theSuccessProb*100, "%\n")
	itcurrent+= 1
	print("\t sous-iter=", itcurrent)
	## - loop until we have at least 10 sets of samples
	if itcurrent < 10:
		ctx.dat= itcurrent
		if itcurrent == 1:
			ctx.probaDist= [theSuccessProb]
		else:
			ctx.probaDist.append(theSuccessProb)
		return -1, theSuccessProb
	else:
		## - Otherwise, set the condition for next iteration of the Lin2Bench
		ctx.dat= 0
		mymean= np.mean(ctx.probaDist)
		myvar= np.std(ctx.probaDist, ddof=1)
		ctx.distVectors[0].append(ncurrent)
		ctx.distVectors[1].append(mymean)
		ctx.distVectors[2].append(myvar)
		
		## - test for the end of the benchmark
		print(f"evaLin2Score: M={mymean} sigma={myvar}")
		myret= 1 if (mymean < 0.1) else 0

		if myret== 1 and len(ctx.distVectors[0]) >= 2:
			return myret, fitSigmoid(ctx)
		return myret, (mymean, myvar)

def main():
	r"""
	 Main program: unitary and integration tests (when not used as a library)
	"""
	# Below, samples from Zephyr run 2026
	lx= [ 1 , 6 , 11, 16 , 21, 
		26, 31, 36 , 41, 46 ,
		 51, 56, 61, 66, 71]
	ly= [ 0.9925555555555555, 0.976, 0.979, 0.9565555555555556, 0.896, 
		0.8303333333333333, 0.7828888888888889, 0.616111111111111, 0.5886666666666667, 0.3177777777777778, 
		0.3804444444444444, 0.13811111111111113, 0.10844444444444444, 0.1, 0.04544444444444444]
	ldy= [ 0.016919745204280655, 0.022181073012818825, 0.019615045245933018, 0.028178498501122744, 0.07781066764910838,
		 0.09952261049630882, 0.1733929384695672, 0.2035641449546337, 0.17379010328554387, 0.2385863458885366, 
		 0.2282827802918516, 0.1043101678222747, 0.1173436311768891, 0.12353339629428149, 0.049000283445892204]
	mctx= QPUBenchContext()
	mctx.distVectors= [lx,ly,ldy]
	fitSigmoid(mctx,'lin2bench-test.svg')


if __name__ == "__main__":
	main()
