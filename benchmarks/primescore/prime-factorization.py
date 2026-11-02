#!/bin/python3

import numpy as np

"""
  Update list with new value
"""
def listupdate(mylist, myelement):
	le= len(mylist)
	ind=0
	for listel in mylist:
		# loop at each element of the list
		ind= 0
		tot= len(listel)-1
		for i in range(tot):
			# check if all indexes match
			if listel[i] != myelement[i]:
				# if not go to next element of the list
				break
			ind+= 1
			# if they do
			if ind == tot:
				listel[ind]+= myelement[ind]
				ind= -1
				#print("   >>>>", listel)
				break
		if ind == -1: # element found and updated
			break
	if ind != -1:
		mylist.append(myelement)
	return mylist		
			

"""
  Transform factorization (N-np)^2 into HOBO
  N integer to factorize
  nbits number of bits in first prime
  pbits number of bits in second prime
"""
def HobofyPrime(N, nbits, pbits):
	# initialize the number of variables before quadratization
	nvar= nbits*pbits
	list2= []
	list3= []
	list4= []
	# first develop (N-np)
	myP0= np.zeros((pbits,nbits)) #myP0 is the matrix of the coefficient of the polynom "np" (quadatic, binary, not positive)
	for j in range(nbits):
		for i in range(pbits):
			myP0[i][j]= (1<<i)*(1<<j)
	#print(myP0)
	# now, use this first polynom and square (N-np)
	maxvar= nbits*pbits
	myPol= np.zeros((maxvar,maxvar))
	for i1 in range(-1,nbits):
		if i1 >=0:
			for i2 in range(pbits):
				for j1 in range(nbits):
					for j2 in range(pbits):
						coef1= -myP0[i2][i1]
						coef2= -myP0[j2][j1]
						#if (i1 == j1) and (i2 == j2):
						mul= 1
						#else:
						#	mul= 2
						#print(i1," ",nbits+i2," ",j1," ",nbits+j2," = ",coef1*coef2*mul)
						if i1 == j1 and i2 == j2:
							#list2.append([i1,i2+nbits,coef1*coef2*mul])
							listupdate(list2, [i1,i2+nbits,coef1*coef2*mul])
						elif i1 != j1 and i2 != j2:
							ind1= min(i1,j1)
							ind2= min(i2,j2)
							ind3= max(i1,j1)
							ind4= max(i2,j2)
							#list4.append([ind1,ind3,ind2+nbits,ind4+nbits,coef1*coef2*mul])
							listupdate(list4,[ind1,ind3,ind2+nbits,ind4+nbits,coef1*coef2*mul])
						else:
							if i1 == j1:
								ind1= i1
								ind2= min(i2,j2)+nbits
								ind3= max(i2,j2)+nbits
							else:
								ind1= min(i1,j1)
								ind2= max(i1,j1)
								ind3= i2+nbits
							#list3.append([ind1, ind2, ind3, coef1*coef2*mul])
							listupdate(list3, [ind1, ind2, ind3, coef1*coef2*mul])
		else:
			#print(-1)
			for j1 in range(nbits):
				for j2 in range(pbits):
					coef1= N
					coef2= -myP0[j2][j1]
					#print(-1," ", j1, " ", nbits+j2," = ",N*coef2*2)
					list2.append([j1,nbits+j2,N*coef2*2])
	#print(list2)
	#print(list3)
	#print(list4)
	return list2,list3,list4
	
"""
  Find best config by exhaustive exploration 
"""
def bestConfig(numbit, hobo):
	bestcost= 2^numbit
	bestconf= 0
	for cfg in range(2**numbit):
		cost= 0
		for elmt in hobo:
			partcost= 1
			for num in elmt[:-1]:
				#print("\t\t>>>",1<<num, cfg, ((1<<num)& cfg)>>(num))
				partcost*= ((1<<num) & cfg)>>num # is 0 or 1
				if partcost == 0:
					break
			#print("###",elmt[-1])
			partcost*= elmt[-1]
			cost+= partcost
			#print("\t----",hex(cfg),elmt,partcost)
		#print("   ", hex(cfg), cost)
		if cost < bestcost:
			bestcost= cost
			bestconf= cfg
	print(hex(bestconf), bestcost)
	return bestconf, bestcost
			
	
"""
  Quadratization procedure, input= list of tuples with higher order factors, number of initial variables, output= list of quadritized terms
"""
def quadratize(nvar, initlist):
	k=0
	indexi= -1
	indexj= -1
	#print(initlist)
	diclist= []
	newlist2= []
	curvar= nvar
	while len(initlist) > 0:
		listel= initlist[0]
		listlen= len(listel)-1
		k= 1
		# first, select a couple for quadratization
		mycouple= []
		maxnumfit= 0
		# each possible couple
		for i in range(listlen):
			for j in range(i+1, listlen):
				# each remaining tuple of the list
				#print(listel[i],listel[j])
				numfit= 0
				for explel in initlist[k:]:
					try:
						indexi= explel.index(listel[i])
					except:
						indexi= -1
					finally:
						try:
							indexj= explel.index(listel[j])
						except:
							indexj= -1
						finally:
							if indexi >= 0 and indexj >= 0:
								numfit+= 1
								#print("   >>",explel, indexi, indexj, numfit)
				if numfit > maxnumfit:
					maxnumfit= numfit
					mycouple= [listel[i],listel[j]]
					#print("   selected:",mycouple)
		if len(mycouple) >= 2:
			print("# ancilla for ",mycouple, "is",curvar)
		if len(mycouple) != 2:
			break
		diclist= diclist + mycouple
		# Now replace occurences in initlist with new couple
		# Starting with the current
		if len(listel) == 4:
			# remove element from list and append new to newlist2
			for i in listel:
				if listel != mycouple[0] and listel != mycouple[1]:
					#found
					newlist2= newlist2 + [[i,curvar,listel[3]]]
					# Rosenberg
					newlist2= newlist2 + [mycouple + ["M"]]
					newlist2= newlist2 + [[mycouple[0],curvar,"2M"]]
					newlist2= newlist2 + [[mycouple[1],curvar,"2M"]]
					newlist2= newlist2 + [[curvar, "3M"]]
					initlist.pop(0)
					break
		else:
			newlistitem= []
			for i in listel[:-1]:
				if i != mycouple[0] and i != mycouple[1]:
					newlist= newlist + [i]
					#print(" +",newlist)
			newlist= newlist + [curvar, listel[-1]]
			#print(" += ", newlist)
			initlist.pop(0)
			initlist.insert(0,newlist)
		# then the remaining
		for indlist in range(k,len(initlist)):
			if indlist >= len(initlist):
				break
			explel= initlist[indlist]
			#print(explel)
			newtuple= []
			try:
				indexi= explel.index(mycouple[0])
			except:
				indexi= -1
			finally:
				try:
					indexj= explel.index(mycouple[1])
				except:
					indexj= -1
				finally:
					if indexi >= 0 and indexj >= 0:
						#another one fit
						for i in explel:
							if i != mycouple[0] and i != mycouple[1]:	
								if type(i) == int:
									newtuple= newtuple + [i]
								else:
									newtuple= newtuple + [curvar, i]
						if len(newtuple) == 3:
							initlist.remove(explel)
							newlist2= newlist2 + [newtuple]
							indlist-= 1
							if indlist >= len(initlist):
								break
						elif len(newtuple) <3:
							print("$$$$$$ Fatal error $$$$")
						else:
							initlist.pop(indlist)
							initlist.insert(indlist,newtuple)
			#print("   >>",newtuple)
		curvar+= 1
	#print(curvar,"--",newlist2)
	return curvar,newlist2

"""
Replace M value for Rosenberg quadratization
 New factor is: M(x_i x_j - 2 x_i Q_{ij} -2 x_j Q_{ij} + 3 Q_{ij})
"""
def defMval(mylist, myM):
	for i in mylist:
		#print(type(i[-1]))
		if type(i[-1]) == str or type(i[-1]) == list:
			if i[-1] == "3M":
				i[-1]= 3*myM
			elif i[-1] == "2M":
				i[-1]= -2*myM
			else:
				i[-1]= myM
	return mylist
	
"""
Reduce redundant entries in the polynomial expression (dinomial only)
"""
def reducePolynomial(numvar,myDinom):
	mat= np.zeros( (numvar,numvar) )
	for el in myDinom:
		print("    ",el)
		if len(el)==3:
			i= el[0]
			j= el[1]
			val= el[2]
		elif len(el) == 2:
			i= el[0]
			j= i
			val= el[1]
		else:
			assert(False)
		if j < i:
			swap=i
			i=j
			j=swap
		mat[i][j]+= val
	outlist= []
	print("Qubo matrix:\n",mat)
	for i in range(numvar):
		for j in range(i,numvar):
			if abs(mat[i][j]) > 1e-8: #only non zero values
				if i != j:
					outlist= outlist + [ [i,j,mat[i,j]] ]
				else:
					outlist= outlist + [ [i,mat[i,j]] ]
	return outlist 
		
	
"""
Output dictionnaries for QUBO computing on the QPU
"""
def exportQUBOdict(mylist):
	hlist= "h={"
	hini= False
	Jlist= "J={"
	Jini= False
	for i in mylist:
		if len(i) > 2:
			if Jini:
				Jlist= Jlist+","
			else:
				Jini= True
			Jlist= Jlist+"({},{}):{}".format(i[0],i[1],i[2])
		else:
			if hini:
				hlist= hlist+","
			else:
				hini= True
			hlist= hlist+"({}):{}".format(i[0],i[1])
	hlist= hlist+"}"
	Jlist= Jlist+"}"
	return hlist,Jlist

tofactorize= 15
mylist2,mylist3,mylist4= HobofyPrime(tofactorize,3,2)
print("HOBO for ",tofactorize," is:")
print(mylist2)
print(mylist3)
print(mylist4)
print("Best configuration and cost:")
bestConfig(5,mylist2+mylist3+mylist4)
print("\n\nQuadratization:")
list34= mylist3+mylist4
totvar,newlist2= quadratize(5,list34)
print(newlist2)
print("Number of variables after quadratization:",totvar)
#valuedlist=defMval(newlist2,176.0)
valuedlist=defMval(newlist2,200.0)

print("Best quadratization cost:")
bestConfig(totvar, mylist2+valuedlist)
finalList2= reducePolynomial(totvar,mylist2+valuedlist)
myh,myJ= exportQUBOdict(finalList2)
print("")
print(myh)
print(myJ)
