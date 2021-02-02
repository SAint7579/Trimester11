#Importing the libraries
import numpy as np

def Union(set1,set2):
	''' To calculate the Union'''
	ret_dict = dict()
	for x, y in zip(set1,set2):
		#Taking the max
		if set1[x] > set2[y]:
			ret_dict[x] = set1[x]
		else:
			ret_dict[y] = set2[y]
	print("\n\nUnion ",ret_dict)
	
def Inter(set1,set2):
	''' To calculate the Intersection'''
	ret_dict = dict()
	for x, y in zip(set1,set2):
		#Taking the min
		if set1[x] < set2[y]:
			ret_dict[x] = set1[x]
		else:
			ret_dict[y] = set2[y]
	print("\n\nIntersection ",ret_dict)
			
def Diff(set1,set2):
	''' To calculate the Difference'''
	ret_dict = dict()
	for x, y in zip(set1,set2):
		#Calculating the complement of set 2
		comp = 1 - set2[y]
		#Finding the minimum
		if set1[x] < comp:
			ret_dict[x] = set1[x]
		else:
			ret_dict[y] = comp
	print("\n\nDifference ",ret_dict)

def Comp(set):
	''' To calculate the Complement'''
	ret_dict = dict()
	for k in set.keys():
		##Appending the complement
		ret_dict[k] = 1 - set[k]
	print("\n\nComplement ",ret_dict)
		
		
def CartisianProduct(set1,set2):
	''' To calculate the cartisian product'''
	ret_array = []
	#Finding the cartisian product
	for i in set1.keys():
		for j in set2.keys():
			#Appending the pair and min
			ret_array.append([i,j,min(set1[i],set2[j])])
	
	print("\n\nCartisian Product ", ret_array)
	
	#Matix Format for the product
	dimension = len(set2)
	mat = []
	temp = []
	for i in range(len(ret_array)):
		temp.append(ret_array[i][2])
		#Adding a new row
		if (i+1)%dimension == 0:
			mat.append(temp)
			temp = []
			
	return ret_array,np.array(mat)
		
def MaxMin_Composition(r1,r2):
	''' To calculate the MaxMin Composition'''
	ret = []
	for x1 in r1:
		for y1 in r2.T:
			#Adding the max min value
			ret.append(max(np.minimum(x1, y1)))

	print("\n\nMaxMin Composition\n", np.array(ret).reshape((r1.shape[0], r2.shape[1])))
	return np.array(ret).reshape((r1.shape[0], r2.shape[1]))
	
	
if __name__ == "__main__":
	A = {"l": 0.4, "m": 0.3, "n": 0.2, "o": 0.1} 
	B = {"l": 0.1, "m": 0.5, "n": 0.2, "o": 0.9} 
	
	print('The First Fuzzy Set is :', A) 
	print('The Second Fuzzy Set is :', B) 
	
	#Union
	Union(A,B)
	
	#Intersection
	Inter(A,B)
	
	#Difference
	Diff(A,B)
	
	#Complement
	Comp(A)
	
	#Cartisian Product
	Product1,mat1 = CartisianProduct(A,B)
	
	C = {"a": 0.1, "b": 0.5, "c": 0.2, "d": 0.9} 
	
	print('\n\nFuzzy Set for Second Product:', C) 
	Product2,mat2 = CartisianProduct(B,C)
	
	
	#MaxMin Composition
	MaxMin_Composition(mat1,mat2)

	
'''
The First Fuzzy Set is : {'l': 0.4, 'm': 0.3, 'n': 0.2, 'o': 0.1}
The Second Fuzzy Set is : {'l': 0.1, 'm': 0.5, 'n': 0.2, 'o': 0.9}


Union  {'l': 0.4, 'm': 0.5, 'n': 0.2, 'o': 0.9}


Intersection  {'l': 0.1, 'm': 0.3, 'n': 0.2, 'o': 0.1}


Difference  {'l': 0.4, 'm': 0.3, 'n': 0.2, 'o': 0.09999999999999998}


Complement  {'l': 0.6, 'm': 0.7, 'n': 0.8, 'o': 0.9}


Cartisian Product  [['l', 'l', 0.1], ['l', 'm', 0.4], ['l', 'n', 0.2], ['l', 'o', 0.4], ['m', 'l', 0.1], ['m', 'm', 0.3], ['m', 'n', 0.2], ['m', 'o', 0.3], ['n', 'l', 0.1], ['n', 'm', 0.2], ['n', 'n', 0.2], ['n', 'o', 0.2], ['o', 'l', 0.1], ['o', 'm', 0.1], ['o', 'n', 0.1], ['o', 'o', 0.1]]


Fuzzy Set for Second Product: {'a': 0.1, 'b': 0.5, 'c': 0.2, 'd': 0.9}


Cartisian Product  [['l', 'a', 0.1], ['l', 'b', 0.1], ['l', 'c', 0.1], ['l', 'd', 0.1], ['m', 'a', 0.1], ['m', 'b', 0.5], ['m', 'c', 0.2], ['m', 'd', 0.5], ['n', 'a', 0.1], ['n', 'b', 0.2], ['n', 'c', 0.2], ['n', 'd', 0.2], ['o', 'a', 0.1], ['o', 'b', 0.5], ['o', 'c', 0.2], ['o', 'd', 0.9]]


MaxMin Composition
 [[0.1 0.4 0.2 0.4]
 [0.1 0.3 0.2 0.3]
 [0.1 0.2 0.2 0.2]
 [0.1 0.1 0.1 0.1]]

'''