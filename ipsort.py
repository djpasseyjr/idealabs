import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
import random
from matplotlib import pyplot as plt

def ipsort(A, sortRow=True,returnPermu=False,verbose=False):
    """ Group similar rows and columns of a matrix together

        Parameters
        ----------
        A: (mxn ndarray) sparse matrix of zeros and ones to sort
        sortRow: (bool) if sortRow=false, the algorithm sorts columns ONLY
        returnPermu: (bool) if returnPermu=True, the function returns the 
                   permutation vectors and grouping vectors
        verbose: (bool) if verbose=True, the function prints it's progress 
                        through the matrix
        Returns
        -------
        Asorted: (mxn ndarray) permuted A matrix
        perRow = (1xm ndarray) row permutations
        perCol (1xn ndarray) column permutations
        rowGroups = (list) stores the indices that begin each row grouping
        colGroups = (list) stores the indices that begin each row grouping
    """
    #Variables to keep track of progress
    M,N = A.shape
    mark = 0
    progress = 0
    printStepSize = N//20
    
    #Variables to store row and column permutations
    perCol = np.arange(N) 
    perRow = np.arange(M) 
    
    #Variables to store the beginnings of each grouping
    rowGroups = [0] 
    colGroups = [0]
    
    A = A.tocsc()
    while mark < N-2:
        #Compute column similarity with a dot product
        scores = A[:,mark].transpose().dot(A[:,mark+1:])
        
        #Move the most similar columns to the front
        sort = np.argsort(scores.toarray().ravel())[::-1] 
        permu = list(range(mark+1)+list(sort+mark+1))
        A = A[:,permu]
        perCol = perCol[permu]
        
        #Find the end of the grouping
        fin = np.argmax(np.abs(np.diff(scores.toarray()[:,sort])))
        mark += fin + 2
        colGroups.append(mark)
        
        if verbose:
            if mark - progress > printStepSize:
                steps = progress//printStepSize
                print("Column progress: \t{}".format('%'*steps) + '-'*(20-steps-2) +'|')
                progress += printStepSize
            
        
    #Repeat the above algorithm on the transposed matrix
    if sortRow:
        mark = 0
        progress = 0
        printStepSize = M//20
        
        A = A.transpose()#Transpose and run the algorithm above
        
        while mark < N-2:
            #Compute column similarity with a dot product
            scores = A[:,mark].transpose().dot(A[:,mark+1:])

            #Move the most similar columns to the front
            sort = np.argsort(scores.toarray().ravel())[::-1] 
            permu = list(range(mark+1)+list(sort+mark+1))
            A = A[:,permu]
            perRow = perRow[permu]

            #Find the end of the grouping
            fin = np.argmax(np.abs(np.diff(scores.toarray()[:,sort])))
            mark += fin + 2
            rowGroups.append(mark)
            
            if verbose:
                if mark - progress > printStepSize:
                    steps = progress//printStepSize
                    print("Row progress: \t\t{}".format('%'*steps) + '-'*(20-steps-2) +'|')
                    progress += printStepSize
                    
        A = A.transpose() #Transpose it back

    if returnPermu:
        return A,perRow,perCol,rowGroups,colGroups
    return A
    
def plot_coo_matrix(m):
    """ Plots nonzero entries of sparse array m as black pixels
        Credit to:
        https://stackoverflow.com/questions/22961541/
        python-matplotlib-plot-sparse-matrix-pattern
    """

    if not isinstance(m, sparse.coo_matrix):
        m = sparse.coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='white')
    ax.plot(m.col, m.row, marker=',', color='black', lw=0,linestyle="")
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def shuffSparseArray(M=100,N=100):
    """Returns a sparse array with random clusters"""

    A = sparse.dok_matrix((M**2,N**2))
    secureRandom = random.SystemRandom()
    unusedCol = list(np.arange(M**2)) #List of empty columns
   
    while len(unusedCol)!=0:

        #Form a column with three 1s placed randomly at entryInd
        entryInd = np.random.randint(M**2,size=int(M/3))
        
        #Choose R random places to put this col 
        R = np.random.randint(N/2,high=N)
        if len(unusedCol) < N: 
            R = len(unusedCol)
    
        for i in range(R):
            col = secureRandom.choice(unusedCol)
            unusedCol.remove(col)
            for j in range(len(entryInd)):
                A[entryInd[j],col] = 1

    return A        
