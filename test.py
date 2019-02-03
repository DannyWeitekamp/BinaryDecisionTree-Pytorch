import torch
from pprint import pprint


class Node(object):
	def __init__(self, split_on, left_child,right_child):
		"""
		The tree constructor.
		"""
		self.split_on = split_on
		self.left_child = left_child
		self.right_child = right_child

		#toDO ADD PARENTS
		self.parent = None
		self.counts = None

	def __str__(self,level=0,delim='   '):
		num = "*" if self.split_on == 0 else str(self.split_on)
		d = " ".join([chr(a+64)+":"+str(b) for a,b in sorted(self.counts.items(),key=lambda x: x[0])])
		ret = delim*level+num + ": " + d +"\n"
		# ret = delim*level+num + ": " + str({chr(a+65):b for a,b in self.counts.items()})+"\n"

		if(self.split_on == 0): return ret
		for child in [self.left_child,self.right_child]:
			if(child != None):
				ret += child.__str__(level=level+1)
			else:
				ret += delim * level + child.__str__() + "\n"
		return ret
		
		# self.counts
		

def counts_per_split(x,y,u):
	'''
	x: (n.d)
	y: (n,)
	
	'''
	n = x.size()[0]
	

	y = y.unsqueeze(1)

	n_x = ~x

	# print(x.dtype, n_x.dtype)

	X = torch.cat((x,n_x),dim=1)
	# print("x",X.size(), X.dtype)
	# print("y", y.size(), y.dtype)
	mul = X * y #could also be torch.where(X,y,X)

	print("Mul")
	print(mul)
	# print(mul.size())

	# counts = torch.bincount(mul,minlength=num_bins)

	# ar = torch.arange(num_bins,dtype=torch.uint8).view(num_bins,1,1)+1

	# print(ar)
	print(u)

	eq = torch.eq(mul, u.view(u.size()[0],1,1))
# 	
	# print("eq")
	# print(eq)
	# print(eq.size())


	counts = torch.transpose(torch.sum(eq,1,dtype=torch.float).type(torch.uint8),0,1)

	# print("counts")
	# print(counts)
	# print(counts.size())
	return counts

def gini(counts):
	'''
	counts: (s_n, c_n)

	 '''

	totals = torch.sum(counts,1,keepdim=True, dtype=torch.float) + 1e-10 #epsilon

	# print("totals")
	# print(totals)


	prob = counts.type(dtype=torch.float)/totals #Sum over the instances

	# print("prob")
	# print(prob)
	# print(prob.size())

	gini_impurities = (1-torch.sum((prob * prob),1))

	return gini_impurities

	# counts = torch.bincount(y,minlength=num_bins)



	# eq = torch.eq(y, torch.arange(num_bins))
def select_all_except(x, dim, execpt):
	d = x.size()[dim]
	return torch.index_select(x,dim,torch.cat(
		(torch.arange(execpt), torch.arange(d-(execpt+1))+execpt+1 )
		))


def split_tree(x,y, entropy, u):
	# if(y_map == None): y_map = torch.arange(num_bins+1)


	d = x.size()[1]
	counts = counts_per_split(x,y,u)

	# print(counts)

	entropies = gini(counts).view(2,d)
	information_gains = entropy - entropies.sum(0)
	max_split = torch.argmax(information_gains)

	if(information_gains[max_split].data.item() > 0):
		f_max_split = max_split.data.item()
		mask = x[:,f_max_split]
		split_entropies = entropies[:,f_max_split]

		if(split_entropies[0] > 0 or split_entropies[1] > 0):
			sub_x = select_all_except(x,1,f_max_split)
			if(split_entropies[0] > 0):
				right_selction = torch.masked_select(torch.arange(x.size()[0]) , mask)	
				right_x = torch.index_select(sub_x, 0, right_selction)
				right_y = torch.index_select(y, 0, right_selction)

				print("right")
				# print(right_y, right_y.dtype)

				right_u, right_y_mapped = torch.unique(right_y, return_inverse=True)
				# right_y = (right_y_mapped+1).type(dtype=torch.uint8)

				# right_y_map = y_map[unique]

				# print("right")
				# print(right_x, right_x.dtype)
				print(right_u)
				# print(right_y, right_y.dtype)

				right_split = split_tree(right_x,right_y,split_entropies[0],right_u)

				
				# print(right_split)
				# if(len(right_split) > 0 and right_split[0] >= f_max_split): right_split[0] += 1
			else:
				right_split = Node(0,None,None)



			if(split_entropies[1] > 0):
				left_selction = torch.masked_select(torch.arange(x.size()[0]) , ~mask)
				left_x = torch.index_select(sub_x, 0, left_selction)
				left_y = torch.index_select(y, 0, left_selction)

				print("left")
				# print(left_y, left_y.dtype)
				# u = torch.unique(left_y)
				# left_y_mapped = torch.arange(len(u),dtype=torch.uint8)+1
				left_u, left_y_mapped = torch.unique(left_y, return_inverse=True)
				# left_y = (left_y_mapped+1).type(dtype=torch.uint8)
# 
				print(left_u)
				# print(left_y, left_y.dtype)
				# print("left")
				# print(left_x, left_x.dtype)
				# print(left_y, left_y.dtype)

				left_split = split_tree(left_x,left_y,split_entropies[1],left_u)
				# left_split.counts = counts[f_max_split + d]
				# print(f_max_split,left_split)
				# if(len(left_split) > 0 and left_split[0] >= f_max_split): left_split[0] += 1
			else:
				left_split = Node(0,None,None)
			
		else:
			left_split, right_split = Node(0,None,None), Node(0,None,None)
				
		right_split.counts = {a:b for a,b in zip(u.tolist(), counts[f_max_split].tolist())}
		left_split.counts = {a:b for a,b in zip(u.tolist(), counts[f_max_split + d].tolist())}

		return Node(f_max_split,left_split,right_split)
	else:
		return None
	

def fix_splits(node,diff=1):
	if(node.left_child != None):
		if(node.left_child.split_on + diff > node.split_on):
			node.left_child.split_on += diff
		fix_splits(node.left_child,diff+1)

	if(node.right_child != None):
		if(node.right_child.split_on >= node.split_on): node.right_child.split_on += 1
		fix_splits(node.right_child,diff+1)

def binary_decision_tree(x,y):

	u = torch.unique(y)
	# c_n = torch.sum(torch.nonzero())

	counts = torch.bincount(y)[1:].unsqueeze(0)


	entropy = gini(counts)


	counts_list = counts.tolist()[0]
	counts_dict = {a:b for a,b in zip(range(1,len(counts_list)+1), counts_list )}

	# print("START ET", entropy)

	# print("y coutns", counts)

	# entropy = 0#gini(counts)

	print(x.dtype, y.dtype)
	splits = split_tree(x,y,entropy,u)
	splits.counts = counts_dict
	fix_splits(splits)




	# print("OUT", splits)
	print(splits)
	print(counts.tolist()[0])

	# print(eq)

	# print(counts)


bloo = torch.tensor(
	[[1,1,0,1,1],
	 [1,0,1,1,1],
	 [0,1,0,0,1],
	 [0,1,0,1,0]],
	 dtype=torch.uint8)

blehh = torch.tensor([1,1,2,3],
	dtype=torch.uint8)

binary_decision_tree(bloo,blehh)


import numpy as np
# x = torch.tensor(np.random.randint(low=0,high=2,size=(200,100)),dtype=torch.uint8)
# y = torch.tensor(np.random.randint(low=1,high=50,size=(200,)),dtype=torch.uint8)

# x = torch.tensor(np.eye(10),dtype=torch.uint8)
# y = torch.tensor(np.arange(10)+1,dtype=torch.uint8)

# print(x)
# print(y)
# binary_decision_tree(x,y)


# x = torch.tensor((np.eye(10)-1)*-1,dtype=torch.uint8)
# y = torch.tensor(np.arange(10)+1,dtype=torch.uint8)

# print(x)
# print(y)
# binary_decision_tree(x,y)


bloo = torch.tensor(
	[
	[0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],
	[0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,0,0],
	[0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0],
	[1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,1],
	[0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0],
	[1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0],
	[0,0,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1],
	],
	 dtype=torch.uint8)

blehh = torch.tensor([1,1,1,2,2,2,3],
	dtype=torch.uint8)

binary_decision_tree(bloo,blehh)