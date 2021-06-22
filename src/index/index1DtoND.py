################################################################################################################################
# Take a 1D index and return indices for each dimension of the wave function (x/y/z for each particle)
################################################################################################################################

def index(ind, inds, sizes, dim, p = 1): 

    l = p * dim
    N = 1
    for i in range(0,dim):
        N *= sizes[i]
        
    N = N ** p
    stride = N
    for i in range(0,l):
        temp = stride
        stride = temp // sizes[i % dim]
        inds[i] = ind // stride
        ind = ind - inds[i] * stride
        
    
