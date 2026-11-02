# set of equations given as Ax+B=0 (A matrix n×n and B vector)
# using unsigned (sig=1) or signed (sig=-1) conventions
# numbit is the number of required bits for the resolution
import numpy as np

"""
Positive 3 bits/variable, 3 var
"""
def eq3_3_u (A,B,numbit,sig):
	#A,B,numbit,sig
	A= np.array([[2,1,-1],[1,2,-1],[1,1,1]]) # solution is 1,3,5
	B= np.array([[0],[-2],[-9]])
	sig=1
	numbit=3
	return A,B,numbit,sig


"""
Positive 2 bits/variable, 2 var
"""
def eq2_2_u (A,B,numbit,sig):
	#global A,B,s,numbit
	A= np.array([[1,-1],[2,-1]])
	B= np.array([[1],[-1]])
	sig=1
	numbit=2
	return A,B,numbit,sig

"""
Negative 2 bits/variable, 2 var
"""
def eq2_2_s (A,B,numbit,sig):
	#global A,B,s,numbit
	A= np.array([[2,1],[1,-1]])
	B= np.array([[0],[-3]])
	sig=-1
	numbit= 2
	return A,B,numbit,sig

"""
Positive 3 bits/variable (6 and 5), 2 var
"""
def eq2_3_u (A,B,numbit,sig):
	#global A,B,s,numbit
	A= np.array([[1,-2],[1,-1]])
	B= np.array([[4],[-1]])
	sig=1
	numbit=3
	return A,B,numbit,sig

"""
Negative 3 bits/variable (-3 2), 2 var
"""
def eq2_3_s (A,B,numbit,sig):
	#global A,B,s,numbit
	A= np.array([[1,2],[2,3]])
	B= np.array([[-1],[0]])
	sig=-1
	numbit=3
	return A,B,numbit,sig

"""
Positive 2 bits/variable, 3 var
"""
def eq3_2_u (A,B,numbit,sig):
	#A,B,numbit,sig
	A= np.array([[1,-1,+1],[2,1,-2],[3,1,-5]]) # solution is 1,3,2
	B= np.array([[0],[-1],[+4]])
	sig=1
	numbit=2
	return A,B,numbit,sig

"""
Positive 2x2 bits/variable, 3 var
"""
def eq3_2_2_1_u (A,B,numbit,sig):
	#A,B,numbit,sig
	A= np.array([[1,-4,+1],[2,1,-2],[3,1,-5]]) # solution is 1,3,2
	B= np.array([[0],[-1],[+4]])
	sig=1
	numbit=2
	return A,B,numbit,sig

"""
G_1 matrix
"""
def g_1_qubo(Q):
	Q= np.array([[-17,0,16,16,0,0,0,0],\
		[0,-17,0,0,16,16,0,0],\
		[0,0,-17,16,16,0,16,0],\
		[0,0,0,-17,0,16,0,16],\
		[0,0,0,0,-17,16,16,0],\
		[0,0,0,0,0,-17,0,16],\
		[0,0,0,0,0,0,-17,0],\
		[0,0,0,0,0,0,0,-17]])
	#h= np.array([[-17],[-17],[-17],[-17],[-17],[-17],[-17],[-17]])
	return Q

"""
G_1_simp matrix
"""
def g_1_simple_qubo(Q):
	Q= np.array([[-1,0,1,1,0,0,0,0],\
		[0,-1,0,0,1,1,0,0],\
		[0,0,-1,1,1,0,1,0],\
		[0,0,0,-1,0,1,0,1],\
		[0,0,0,0,-1,1,1,0],\
		[0,0,0,0,0,-1,0,1],\
		[0,0,0,0,0,0,-1,0],\
		[0,0,0,0,0,0,0,-1]])
	#h= np.array([[-17],[-17],[-17],[-17],[-17],[-17],[-17],[-17]])
	return Q

