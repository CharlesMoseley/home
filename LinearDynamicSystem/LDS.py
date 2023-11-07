import argparse
import numpy as np
from scipy.linalg import pinv, svd # Your only additional allowed imports!

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Homework 2",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2023",
        add_help = "How to use",
        prog = "python homework2.py <arguments>")
    parser.add_argument("-f", "--infile", required = True,
        help = "Dynamic texture file, a NumPy array.")
    parser.add_argument("-q", "--dimensions", required = True, type = int,
        help = "Number of state-space dimensions to use.")
    parser.add_argument("-o", "--output", required = True,
        help = "Path where the 1-step prediction will be saved as a NumPy array.")

    args = vars(parser.parse_args())

    # Collect the arguments.
    input_file = args['infile']
    q = args['dimensions'] 
    output_file = args['output']

    # Read in the dynamic texture data.
M = np.load(input_file)
    
    ### FINISH ME

f= M.shape[0]
h = M.shape[1]
w = M.shape[2]

Y = np.zeros([h * w, f])


# i in range(f):
 #   flat = M[i,:,:].flatten()
  #  Y = np.insert(Y,i,flat,axis=1)
#Y = Y[:,0:f]

for i in range(f):
    Y[:,i] = M[i,:,:].flatten()

U, s, Vh = svd(Y)

C = U[:,0:q]
#print(C.shape)
qs = s[0:q]
#print(qs.shape)
sig = np.diagflat(qs)
#print(sig.shape)
V = Vh.T[:,0:q]
#print(V.shape)

X = sig@V.T

X1 = X[:,0:f-2]

X2 = X[:,1:f-1]

X3 = X[:,2:f]


X1i = pinv(X1)
X2i = pinv(X2)

I=np.identity(f-2)

A2 = X3@(I-X2i@X2)@pinv(X1@(I-X2i@X2))
A1 = X3@(I-X1i@X1)@pinv(X2@(I-X1i@X1))

x71 = A1@X[:,f-1] + A2@X[:,f-2]
#x71 = A@X[:,f-1]
#print(x71.shape)
y71 = C@x71
#print(y71.shape)

Y71 = y71.reshape(h,w)

np.save(output_file, Y71)