################################################################################################################################
# Take a  indices for each dimension of the wave function (x/y/z for each particle) and return a 1D index
################################################################################################################################

def index(inds, sizes, dim, p = 0):
    if (p == 0):
        p = params["Np"]

    indOut = 0
    l = p * dim
    for i in range(0, l):
        stride = 1
        for j in range( i + 1, l):
            stride *= sizes[j %  dim]

        indOut += inds[i] * stride
        
    return indOut
