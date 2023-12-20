import numpy as np
import neurolab as nl

# Г С В
target = [[1, 1, 1, 1, 1,
           1, 0, 0, 0, 0,
           1, 0, 0, 0, 0,
           1, 0, 0, 0, 0,
           1, 0, 0, 0, 0],
          [0, 1, 1, 1, 0,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 0,
           1, 0, 0, 0, 1,
           0, 1, 1, 1, 0],
          [1, 1, 1, 1, 0,
           1, 0, 0, 1, 0,
           1, 1, 1, 1, 0,
           1, 0, 0, 1, 0,
           1, 1, 1, 1, 0]]

chars = ['Г', 'С', 'В']
target = np.asfarray(target)
target[target == 0] = -1

# Create and train network
net = nl.net.newhop(target)

output = net.sim(target)
print("Test on train samples:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())

##########################################

print("\nTest on defaced 'Г':")
test_g = np.asfarray([0, 1, 1, 1, 1,
                      1, 0, 0, 0, 0,
                      1, 0, 0, 0, 0,
                      1, 0, 0, 0, 0,
                      1, 0, 0, 0, 0])
test_g[test_g == 0] = -1
out_g = net.sim([test_g])
print((out_g[0] == target[0]).all(), 'Sim. steps', len(net.layers[0].outs))

##########################################

print("\nTest on defaced 'С':")
test = np.asfarray([0, 1, 1, 1, 0,
                    1, 0, 0, 0, 1,
                    1, 0, 0, 0, 0,
                    1, 0, 0, 0, 1,
                    0, 1, 1, 1, 0])
test[test == 0] = -1
out = net.sim([test])
print((out[0] == target[0]).all(), 'Sim. steps', len(net.layers[0].outs))
