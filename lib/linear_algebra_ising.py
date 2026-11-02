#!/bin/python3

import numpy as np
import sys


A,B,numbit,sig= np.array([[1,-1],[2,-1]]), np.array([[1],[-1]]), 2, 1
Qg= np.array([[1,0],[0,-1]])
#sig=1
#numbit=2

## transform Ax+B=0 linear system into a QUBO cost function
def QUBOfy(Amat,Bvect,bitsize,sig=1):
	myl= len(Amat)
	myQ= np.zeros((myl*bitsize,myl*bitsize))
	toto= np.asmatrix(myQ)
	for l in range(myl):
		for i in range(myl):
			#print(Amat[i],"\n ",myQ)
			for j in range(myl):
				for k in range(bitsize):
					for m in range(bitsize):
						# first operate the square product of each line of equation with Amat
						indexQi= i*bitsize+k
						indexQj= j*bitsize+m
						loctot= Amat[l][i]*Amat[l][j]*(1<<k)*(1<<m)
						if(k==bitsize-1): 
							loctot=sig*loctot
						if(m==bitsize-1):
							loctot=sig*loctot
						#print("line ",l," indexes ",indexQi,",",indexQj," value= ", Amat[l][i]," ",Amat[l][j], " total=",loctot)
						if (indexQi<=indexQj):
							#print("Was: ", myQ[indexQi,indexQj])
							myQ[indexQi,indexQj]= myQ[indexQi,indexQj]+ loctot
							#print("Is now: ", myQ[indexQi,indexQj])
						else:
							#print("was: ", myQ[indexQj,indexQi])
							myQ[indexQj,indexQi]= myQ[indexQj,indexQi]+ loctot
							#print("is now: ", myQ[indexQj,indexQi])
			#for each line and each variable and vector element
			loctotmain= Amat[l][i]*Bvect[l][0]
			for k in range(bitsize):
				#and for each bit
				if(k == bitsize-1):
					loctot= sig*2*loctotmain*(1<<k)
				else:
					loctot= 2*loctotmain*(1<<k)
				indexQi= i*bitsize+k
				#print(" index ",indexQi," value=",loctot, " was=",myQ[indexQi,indexQi])
				myQ[indexQi,indexQi]= myQ[indexQi,indexQi]+loctot
				#print("     now=", myQ[indexQi,indexQi])
	return myQ

# Transform QUBO representation to Ising representation
def QUBOtoIsing(myQUBO):
	myl= len(myQUBO)
	myJ= np.zeros((myl,myl))
	toto= np.asmatrix(myJ)
	myH= np.zeros((myl,1))
	toto= np.asmatrix(myH)
	for i in range(myl):
		val= 0.0
		for j in range(myl):
			# set partial J vector
			if(j>i):
				val+= myQUBO[i,j]
				# values are 1/4 of QUBO non diagonal
				# myJ.itemset((i,j),myQUBO[i,j]/4) # numpy v1
				myJ[i,j]= myQUBO[i,j]/4   # this is for numpy v2
			elif (j<i):
				val+= myQUBO[j,i]				
		#H values are a combination of the diagonal elements with non-diagonal
		#print("index: ",i, " val=",val)
		# myH.itemset((i,0),val/4+myQUBO[i,i]/2) # numpy v1
		myH[i,0]= val/4+myQUBO[i,i]/2    # this code is for numpy v2
	return (myH,myJ)

# Transform Ising representation to QUBO representation
def IsingtoQUBO(myH, myJ):
	myl= len(myJ)
	myQ= np.zeros((myl,myl))
	for i in range(myl):
		val= 2.0*myH[i][0]
		for j in range(myl):
			# val is diagonal value of Q and the non diagonal (upper triangle) elements are 4 times the case of QUBO
			if(i<j):
				val-= 2*myJ[i,j]
				myQ[i,j]= myJ[i,j]*4
			elif (i>j):
				val-= 2*myJ[j,i]
		myQ[i,i]= val
	return myQ

# write QUBO matrix for dimod function (D-wave interface to several solvers)	
def QUBOoutput(myQUBO):
	sys.stdout.write("J= { ")
	myl= len(myQUBO)
	for i in range(myl-1):
		for j in range(i+1,myl):
			if(abs(myQUBO[i,j]) > 1e-8):
				sys.stdout.write("("+str(i)+","+str(j)+"):"+str(myQUBO[i,j])+(", " if((i<myl-2)|(j<myl-1)) else ""))
		sys.stdout.write("\\\n")
	sys.stdout.write("}\n")
	sys.stdout.write("h= { ")
	for i in range(myl):
		if(abs(myQUBO[i,i]) > 1e-8):
			sys.stdout.write(str(i)+":"+str(myQUBO[i,i])+(", " if(i<myl-1) else ""))
	sys.stdout.write(" }\n")
	return

# write Python code for D-Wave for dimod function.
def IsingOutput(myH, myJ):
	sys.stdout.write("J= { ")
	myl= len(myJ)
	for i in range(myl-1):
		for j in range(i+1,myl):
			sys.stdout.write("("+str(i)+","+str(j)+"):"+str(myJ[i,j])+(", " if((i<myl-2)|(j<myl-1)) else ""))
		sys.stdout.write("\\\n")
	sys.stdout.write("}\nh= { ")
	for i in range(myl):
		sys.stdout.write(str(i)+":"+str(myH[i][0])+(", " if (i<myl-1) else ""))
	sys.stdout.write(" }\n")
	return
	
# return h descriptor from H vector (Ising model)
def getHforDimod(myh):
	myl= len(myh)
	reth= {}
	for i in range(myl):
		if abs(myh[i][0])> 10e-8:
			reth[i]=myh[i][0]
	return reth
	
# return h descriptor from QUBO matrix for dimod method
def getHQUBOforDimod(myJ):
	myl= len(myJ)
	reth= {}
	for i in range(myl):
		if abs(myJ[i][i]< 10e-8):
			reth[i]= myJ[i][i]
	return reth
	
# return J descriptor from J matrix (Should work both for Ising and QUBO)
def getJforDimod(myJ):
	myl= len(myJ)
	retJ= {}
	for i in range(myl-1):
		for j in range(i+1,myl):
			if abs(myJ[i,j]) > 10e-8:
				retJ[(i,j)]= myJ[i][j]
	return retJ 
	
	
# return Maximum absolute value in the QUBO matrix
def maxValQUBO(myQUBO):
	val= 0.0
	myl= len(myQUBO)
	for i in range(myl):
		for j in range(myl-i):
			curval= abs(myQUBO[i,j+i])
			if(curval> val):
				val= curval
	return val

# return Maximum absolute value in the Ising model
def maxValIsing(myH, myJ):
	myl= len(myH)
	val= 0.0
	for i in range(myl):
		for j in range(myl-i-1):
			curval= abs(myJ[i,j+i+1])
			if(curval>val):
				val=curval
		curval= abs(myH[i][0])
		if(curval>val):
			val= curval
	return val
	
# duplicate QUBO on several logical qubits with /cs/ chain-strenght
def duplicateQUBO(myQUBO, num, cs, wantDiag= True, isLoop=True):
	myl= len(myQUBO)
	myLQ= np.zeros((myl*num,myl*num))
	# first copy /num/ times the base QUBO into myLQ
	for i in range(num):
		for j in range(myl):
			for k in range(myl):
				myLQ[i*myl+j, i*myl+k]= myQUBO[j,k]
	# then implement the logical qubit links, all /num/ of them with /cs/ chain-strenght
	for i in range(num):
		for j in range(num-i-1):
			jindex= j+i+1
			for k in range(myl):
				myLQ[i*myl+k,jindex*myl+k]=-2*cs
		#then the diagonal elements
		if(wantDiag):
			for k in range(myl):
				myvalue= myLQ[i*myl+k,i*myl+k]
				myLQ[i*myl+k,i*myl+k]= myvalue+cs*(num-1)
	return myLQ

# duplicate Ising model by using logical qubits with /cs/ chain strenght
def duplicateIsing(myH, myJ, num, cs, isLoop=True):
	myl= len(myH)
	myLH= np.zeros((myl*num,1))
	myLJ= np.zeros((myl*num,myl*num))
	# copy base model /num/ times
	for i in range(num):
		for j in range(myl):
			for k in range(myl-j):
				myLJ[i*myl+j, i*myl+k+j]= myJ[j,k+j]
			myLH[i*myl+j,0]= myH[j,0]
	# then link the logical qubits with strenght /cs/
	for i in range(num):
		for j in range(num-i-1):
			jindex= j+i+1
			for k in range(myl):
				myLJ[i*myl+k,jindex*myl+k]= -cs
	return (myLH,myLJ)

""" 
Main program
"""

if __name__ == "__main__":
	from mylinearequations import eq3_2_2_1_u as myequation
	from mylinearequations import g_1_qubo as myqubo
	
	# set values from import
	A,B,numbit,sig= myequation(A,B,numbit,sig)
	# create the QUBO matrix
	Qproblem= QUBOfy(A,B,numbit,sig)
	print(Qproblem)
	print(maxValQUBO(Qproblem))
	# transform into an Ising model
	Iproblem= QUBOtoIsing(Qproblem)
	"""
	#print(Iproblem[0]," \n",Iproblem[1])
	"""
	print(maxValIsing(Iproblem[0],Iproblem[1]))
	#display QUBO
	#QUBOoutput(Qproblem)
	#display Ising
	IsingOutput(Iproblem[0],Iproblem[1])
	#second conversion
	Q2= IsingtoQUBO(Iproblem[0],Iproblem[1])
	# display QUBO
	print(Q2)
	# create the logical qubit version of the problem to test if the accuracy increases
	QLp= duplicateQUBO(Qproblem, 2, 0.0, True)
	#print(QLp)
	#QUBOoutput(QLp)
	ILp= duplicateIsing(Iproblem[0],Iproblem[1],3, 18)
	#print(ILp[0],"\n",ILp[1])
	"""
	print
	Qg= myqubo(Qg)
	print(Qg)
	Ig1= QUBOtoIsing(Qg)
	print(Ig1[0]," \n",Ig1[1])
	"""
