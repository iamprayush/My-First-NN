import numpy as np

#each point is [length, width, type(red-1, blue-0)]
data = [[3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, .5,  1],
        [2,   .5,  0],
        [5.5,  1,  1],
        [1,    1,  0]]

mystery_flower = [4.5, 1]
red = 1
blue = 0

#weights and biases
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

def sigmoid(n):
  return 1/(1+np.exp(-n))
  
#derivative of sigmoid  
def sigmoid_p(n):
	return sigmoid(n) * (1 - sigmoid(n))

x = np.linspace(-5, 5, 10)
y = sigmoid(x)


#training loop

l_rate = 0.2

for i in range(100000):
	 #random point from the data
	rand_ind = np.random.randint(len(data))
	point = data[rand_ind]
	
	z = point[0]*w1 + point[1]*w2 + b
	pred = sigmoid(z)
	target = point[2]
	cost = (pred - target) ** 2
	
	#derivative of the cost wrt pred and then pred wrt z and so on...(chain rule)
	dcost_pred = 2 * (pred - target)
	dpred_z = sigmoid_p(z)
	dz_dw1 = point[0]
	dz_dw2 = point[1]
	dz_db = 1


	#ok now, we find deriv of cost(the fn to minimize) wrt to w1, w2, b
	dcost_dw1 = dcost_pred * dpred_z * dz_dw1
	dcost_dw2 = dcost_pred * dpred_z * dz_dw2
	dcost_db = dcost_pred * dpred_z * dz_db

	w1 -= l_rate * dcost_dw1
	w2 -= l_rate * dcost_dw2
	b -= l_rate * dcost_db

	if i % 1000 == 0:
		print("%.3f" %cost)

pred = sigmoid(mystery_flower[0]*w1 + mystery_flower[1]*w1 + b)

if pred > 0.5:
	print("Mystery flower at {}, {} is RED".format(mystery_flower[0], mystery_flower[1]))
else:
	print("Mystery flower at {}, {} is BLUE".format(mystery_flower[0], mystery_flower[1]))
