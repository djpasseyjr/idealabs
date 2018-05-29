from numpy import dot
import numpy.polynomial.polynomial as poly
from copy import deepcopy
from collections import Counter
import control as tf
#import app.control as tf
import numpy as np
from sympy import Matrix
from scipy import linalg as la
import time


# A function to solve (sI-A)^-1
def find_resolvent(A):
    """
    Finds the value of (sI-A)^-1 using Leverrier's Algorithm as explained in:
     On Computing the Inverse of a Rational Matrix by L.H. Keel and S.P. Bhattacharyya
     This method for finding the adjoint and determinant is cited from:
     D.K. Faddeev and V.N. Faddeeva, Computational Methods of LinearAlgebra, Freeman, San Francisco, 1963.
    :param A: A matrix of real numbers
    :return: my_inv: a transfer function which is the resolvent of A ( aka: (sI-A)^-1 )
    """
    n, m = np.shape(A)
    if n != m:
        raise("ValueError: Input matrix, A, must be a square matrix.")
    Ts = [np.eye(n)]
    a_s = [1]
    for i in np.arange(n):
        AT = np.dot(A, Ts[i])
        new_a = -1./(i+1) * np.trace(AT)  # Calculate the next "a"
        a_s.append(new_a)
        # Calculate the next "T"
        new_T = np.dot(A, Ts[i]) + np.eye(n) * new_a
        Ts.append(new_T)
    adj = []
    for i in np.arange(n):
        adj.append([])
        for j in np.arange(n):
            adj[i].append([])
    for i in np.arange(n):
        for j in np.arange(n):
            for k in np.arange(n):
                adj[i][j].append(Ts[k][i, j])
            adj[i][j] = np.array(adj[i][j])
    det = a_s
    # Now we take our inverse as: adj(a matrix)/det(a polynomial)
    # In this application this will translate to making the adjoint our numerator
    # and building a denominator which is a bunch of copies of the determinant.
    den = []
    for i in np.arange(n):
        den.append([])
        for j in np.arange(n):
            den[i].append(np.array(det))
    my_inv = tf.tf(adj, den)
    return my_inv

# Here is a function to invert transfer function matricies
def inverse_rational_matrix(my_tf_object):
    """ Takes the inverse of a transfer function matrix and returns it.
    Parameters:
        numArray_pres: a 2D array of the coefficients of the polynomials in the numerator of each entry of the original matrix
        denArray: a 2D array of the coefficients of the polynomials in the denominator of each entry of the original matrix

    :return:
     out: a transfer function object that represents the inverted matrix
    """
    # We turn our transfer function into a polynomial coefficient type object for making the algebraic inversion
    numArray_reverse = my_tf_object.num
    denArray_reverse = my_tf_object.den
    # Now we take the coefficients of the polynomial and put them into 2 arrays
    # one for numerators and one for denominators
    numArray_list = list(numArray_reverse)
    denArray_list = list(denArray_reverse)
    n = np.shape(denArray_list)[0]
    for i in np.arange(n):
        for j in np.arange(n):
            numArray_list[i][j] = np.flipud(numArray_list[i][j])
            denArray_list[i][j] = np.flipud(denArray_list[i][j])
    numArray_pres = (numArray_list)
    denArray = (denArray_list)
    numArray_fut = deepcopy(numArray_pres)
    our_ks = np.arange(n - 1, -1, -1)
    # We now go through the process of computing the adjoint of the numerator
    for k in our_ks:
        # One iteration of the pivoting, after pivoting the future become the present and the present becomes the past
        # Now we update our past and present and future numerator arrays for the next pivot
        numArray_past = deepcopy(numArray_pres)
        numArray_pres = deepcopy(numArray_fut)
        for i in np.arange(n):
            for j in np.arange(n):
                # Below are the following cases for the numerator of the adjoint,
                # each is based off of the index of the location of the matrix entry:
                # Based off of the paper by T. Downs called: On the inversion of a matrix of rational functions (1971)
                # Case 1: i,j<k
                if i < k and j < k:
                    if k == n-1:
                        numArray_fut[i][j] = tuple(poly.polysub(poly.polymul(poly.polymul(numArray_pres[i][j], numArray_pres[k][k]), poly.polymul(denArray[i][k], denArray[k][j])),
                                                                poly.polymul(poly.polymul(numArray_pres[i][k], numArray_pres[k][j]), poly.polymul(denArray[i][j], denArray[k][k]))))
                    else:
                        numArray_fut[i][j] = tuple(poly.polydiv(poly.polysub(poly.polymul(poly.polymul(numArray_pres[i][j], numArray_pres[k][k]), poly.polymul(denArray[i][k], denArray[k][j])),
                                                                             poly.polymul(poly.polymul(numArray_pres[i][k], numArray_pres[k][j]), poly.polymul(denArray[i][j], denArray[k][k]))),
                                                                             numArray_past[k+1][k+1])[0])
                # Case 2: i>k>j
                elif i > k and k > j:
                    numArray_fut[i][j] = tuple(poly.polydiv(poly.polysub(poly.polymul(poly.polymul(numArray_pres[i][j], numArray_pres[k][k]), denArray[k][j]),
                                                     poly.polymul(poly.polymul(numArray_pres[i][k], numArray_pres[k][j]), denArray[k][k])),
                                                     poly.polymul(denArray[k][i], numArray_past[k+1][k+1]))[0])
                # Case 2.5: j>k>i
                elif j > k and k > i:
                    numArray_fut[i][j] = tuple(poly.polydiv(poly.polysub(poly.polymul(poly.polymul(numArray_pres[i][j], numArray_pres[k][k]), denArray[i][k]),
                                                     poly.polymul(poly.polymul(numArray_pres[i][k], numArray_pres[k][j]), denArray[k][k])),
                                                     poly.polymul(denArray[j][k], numArray_past[k+1][k+1]))[0])
                # Case 4 from the paper: i=j>k
                elif i == j and j > k:
                    numArray_fut[i][j] = tuple(poly.polydiv(poly.polysub(poly.polymul(numArray_pres[i][i], numArray_pres[k][k]),
                                                     poly.polymul(poly.polymul(numArray_pres[i][k], numArray_pres[k][i]), poly.polymul(denArray[k][k], denArray[i][i]))),
                                                     poly.polymul(poly.polymul(denArray[k][i], denArray[i][k]), numArray_past[k+1][k+1]))[0])
                # Case 3 from the paper: i,j>k
                elif i > k and j > k:
                    numArray_fut[i][j] = tuple(poly.polydiv(poly.polysub(poly.polymul(numArray_pres[i][j], numArray_pres[k][k]),
                                                     poly.polymul(poly.polymul(numArray_pres[i][k], numArray_pres[k][j]), denArray[k][k])),
                                                     poly.polymul(poly.polymul(denArray[k][i], denArray[j][k]), numArray_past[k+1][k+1]))[0])
                # Case 5: i=k!=j
                elif i == k and k != j:
                    numArray_fut[k][j] = tuple(poly.polymul((-1,), numArray_pres[k][j]))
                # Case 6: j=k!=i
                elif j == k and k != i:
                    numArray_fut[i][j] = tuple(numArray_pres[i][j])
                # Case 7: i=j=k
                elif i == j == k:
                    if k == n-1:
                        numArray_fut[k][k] = (1,)
                    else:
                        numArray_fut[k][k] = tuple(numArray_past[k+1][k+1])
    denArray_fut = deepcopy(denArray)
    # Below we do the denominator:
    for i in np.arange(n):
        for j in np.arange(n):
            # Each entry of the denominator is the product of all the original denominators except for the ith column
            #  and the jth row of the original denominator matrix
            # First we normalize the initial entry
            denArray_fut[i][j] = tuple(poly.polydiv(denArray_fut[i][j], denArray_fut[i][j])[0])
            for ii in np.arange(n):
                for jj in np.arange(n):
                    if j != ii and i != jj:
                        denArray_fut[i][j] = tuple(poly.polymul(denArray_fut[i][j], denArray[ii][jj]))
    # Now we collect our results and we return our final inverted matrix
    my_adjoint_num = deepcopy(numArray_fut)
    my_adjoint_den = deepcopy(denArray_fut)
    # We multiply the adjoint by the inverse of the determinant of the matrix, the inverse of the determinant is
    # decided to be the past numerator array (or that before preforming the final pivot)
    inverse_determinant_den = deepcopy(numArray_pres[0][0])
    inverse_determinant_num = (1)
    for i in np.arange(n):
        for j in np.arange(n):
            inverse_determinant_num = tuple(poly.polymul(denArray[i][j], inverse_determinant_num))
    newMatrix_num = []
    newMatrix_den = []
    for i in np.arange(n):
        newMatrix_num.append([])
        for j in np.arange(n):
            newMatrix_num[i].append(tuple(poly.polymul(my_adjoint_num[i][j], inverse_determinant_num)))
    # Now we do the denominators
    for i in np.arange(n):
        newMatrix_den.append([])
        for j in np.arange(n):
            x = np.shape(my_adjoint_den)
            newMatrix_den[i].append(tuple(poly.polymul(my_adjoint_den[i][j], inverse_determinant_den)))
    # Ok, we finally have our inverse!
    numerators = newMatrix_num
    denominators = newMatrix_den
    try: # A polynomial solver for smaller degree polynomials to do cancellations and reduce the polynomials
        for i in np.arange(n):
            for j in np.arange(n):
                numerators[i][j] = np.flipud(numerators[i][j])
                denominators[i][j] = np.flipud(denominators[i][j])
                k = 0
                for val in numerators[i][j]:
                    numerators[i][j][k] = round(val, 4)
                    k += 1
                num_roots = (np.roots(numerators[i][j]))
                den_roots = (np.roots(denominators[i][j]))
                for r in np.arange(len(num_roots)):
                    num_roots[r] = round(num_roots[r], 4)*10000.
                for dr in np.arange(len(den_roots)):
                    den_roots[dr] = round(den_roots[dr], 4)*10000.
                num_roots = Counter(num_roots)
                den_roots = Counter(den_roots)
                all_roots_num = num_roots - den_roots
                all_roots_den = den_roots - num_roots
                if np.any(numerators[i][j]) != 0:  # If zero then we want to keep them as they are.
                    numerators[i][j] = numerators[i][j][0]/denominators[i][j][0]
                for k in all_roots_num:
                    for number in np.arange(all_roots_num[k]):
                        numerators[i][j] = list(poly.polymul(numerators[i][j], (-k/10000., 1)))
                denominators[i][j] = 1
                for k in all_roots_den:
                    for number in np.arange(all_roots_den[k]):
                        denominators[i][j] = list(poly.polymul(denominators[i][j], (-k/10000., 1)))
        # Now we return the values in the form of transfer functions by casting them with control.tf
        for i in np.arange(n):
            for j in np.arange(n):
                if type(numerators[i][j]) == 'npumpy.ndarray' or type(numerators[i][j]) == list:
                    numerators[i][j] = np.flipud(numerators[i][j])
                else:
                    numerators[i][j] = np.array([float(np.real(numerators[i][j]))])
                if type(denominators[i][j]) != int:
                    denominators[i][j] = np.flipud(denominators[i][j])
                else:
                    denominators[i][j] = np.array([float(denominators[i][j])])
    except:
        b = "c"
    num_list = []
    den_list = []
    for i in np.arange(n):
        num_list.append(list((numerators[i])))
        den_list.append(list((denominators[i])))
    out = tf.tf(num_list, den_list)
    return out


# this function will help to find the li rows of C, for putting our LTI into a system that is simple to work with
def order_li_rows(A, B, C, D):
    """ Takes the rows of the matrix and reorders them so that the first rows are all of the Linearly Independent ones
    Accepts: A, B, C, D our LTI state space representation of the system
        C a matrix of floats or ints.
    :return:
        LIrows, the LI rows of the matrix, called C_1 in this problem
        C_mod, a matrix of floats or ints with the first rows all LI
    """
    # First we make a copy of the matrix and put it in rref
    C_rref = Matrix(np.transpose(deepcopy(C)))
    # This should return the indicies of the pivot columns of the transpose (AKA, the rows that are LI)
    C_rref = C_rref.rref()[1]
    # We now build a new matrix out of the LI rows and the LD rows and then stack the LI rows on top of the LD rows
    C_mod = np.zeros([1, np.shape(C)[1]])
    C_mod[0:, 0:] = np.array(C[C_rref[0]])
    for index in C_rref[1:]:
        C_mod = np.vstack((C_mod, C[index]))
    LIrows = deepcopy(C_mod)
    # Now we do the LD rows
    LDrows = np.arange(np.shape(C)[0])
    other_index = []
    for index in LDrows:
        if index not in C_rref:
            C_mod = np.vstack((C_mod, C[index]))
            other_index.append(index)
    # If 1D we add another dimension for nullspace calculation
    return A, B, C_mod, D, LIrows


# we define a function that will partition the rows of any given matrix to be used later on in computations
def partition_matrix(matrix, r, c):
    """Takes the matrix, and returns four matricies in a single list, each of them being partitioned along the chosen
     axis, r corresponds to where the index starts in the partition of the rows and c corresponds with the columns."""
    m_11 = matrix[0:r, 0:c]
    m_12 = matrix[0:r, c:]
    m_21 = matrix[r:, 0:c]
    m_22 = matrix[r:, c:]
    return [m_11, m_12, m_21, m_22]


# additionally we will define a function to compute (sI-M)^-1 (meaning the inverse of the discribed matrix M)
def sIminMinv(matrix):
    """ For a square matrix of constants this calculates and returns the matrix: (sI-M)^-1"""
    n = matrix.shape
    if n[0] > 1:
        matrix = -matrix
        tfmatrix = []
        denmatrix = []
        # Below we turn the value of our matrix into something that can be made in to a single transfer function so that
        # We can invert it using our inversion function.
        for i in np.arange(n[0]):
            tfmatrix.append([])
            denmatrix.append([])
            for j in np.arange(n[1]):
                if i == j:
                    if isinstance(matrix[i, j], tf.TransferFunction):
                        a = np.append(0, matrix[i, j].num)
                        b = np.append(matrix[i, j].den, 0)
                        if len(a) < len(b):
                            c = b.copy()
                            c[:len(a)] += a
                        else:
                            c = a.copy()
                            c[:len(b)] += b
                        tfmatrix[i].append(c)
                        denmatrix[i].append(np.append(0, matrix[i, j].den))
                    else:
                        tfmatrix[i].append([1, matrix[i, j]])
                        denmatrix[i].append([1])
                else:
                    if isinstance(matrix[i, j], tf.TransferFunction):
                        tfmatrix[i].append(np.append(0, matrix[i, j].num))
                        denmatrix[i].append(np.append(0, matrix[i, j].den))
                    else:
                        tfmatrix[i].append([matrix[i, j]])
                        denmatrix[i].append([1])
        out = tf.tf(tfmatrix, denmatrix)
        out = inverse_rational_matrix(out)
        return out
    elif np.shape(matrix)[0] == 0:
        out = tf.tf(1, [1., 0])
        out = inverse_rational_matrix(out)
        return out

    # We account for the case in which the input matrix is actually a scalar
    else:
        out = tf.tf([1, -matrix], 1)
        out = inverse_rational_matrix(out)
        return out


# A function for finding the null space of a matrix, used to make our T later on
def nullspace(C):
    """
    calculates a nice basis for the nullspace of C
    :param C: matrix object
    :return: E, the nullspace of C
    """
    rowsC, colsC = np.shape(C)
    # First we augment C with zeros
    zeroes = np.zeros([rowsC, 1])
    C0 = np.hstack([C, zeroes])
    # We put the augmented matrix in rref
    C0 = np.array(Matrix(C0).rref()[0])
    free_cols = []
    for i in np.arange(colsC):
        free_cols.append(False)
    for i in np.arange(colsC):
        for j in np.arange(rowsC):
            if C0[j][i] != 0:
                for k in np.arange(i):
                    if C0[j][k] != 0:
                        free_cols[i] = True
        if np.any(C0[:, i]) == 0:  # The case where the entire column is zero
            free_cols[i] = True
    # For each constrained variable we write it in terms of free variables
    # Each free variable gets its own column
    num_free = 0
    for col in free_cols:
        if col:
            num_free += 1
    E = []
    for i in np.arange(len(free_cols)):
        E.append([])
    col_ind = 0
    row_ind = 0
    for i in np.arange(len(free_cols)):
        if free_cols[i] == False:
            for j in np.arange(colsC):
                if free_cols[j]:
                    E[i].append(-C0[row_ind, j])
            row_ind += 1
        else:
            for j in np.arange(num_free):
                if j == col_ind:
                    E[i].append(1.)
                else:
                    E[i].append(0.)
            col_ind += 1
    return np.array(E)


# A function to turn a constant matrix into a tf
# This function currently handles 0D arrays by making them 1D arrays, this is misleading, however it is accounted
# for in the if else statements around lines 560 and 567 (at the end of find PQR):
def make_matrix_a_tf(matrix):
    """
    Turns a constant matrix into a tf object
    :param matrix: a matrix of constants
    :return: my_tf: a simple transfer function
    """
    # We make zero dim matrices into 1D matrices of zeros
    row_flag = False
    col_flag = False
    dim = list(np.shape(matrix))
    if dim[0] == 0:
        dim[0] = 1
        row_flag = True
    if dim[1] == 0:
        dim[1] = 1
        col_flag = True
    num = []
    den = []
    for i in range(dim[0]):
        num.append([])
        den.append([])
        for j in range(dim[1]):
            if row_flag or col_flag:
                num[i].append(np.array([0.]))
            else:
                num[i].append(np.array([float(matrix[i, j])]))
            den[i].append(np.array([1.]))
    my_tf = tf.tf(num, den)
    return my_tf


# Here is a function to preform MT for matrix M and tf T
def right_mult_matrix_by_tf(matrix, my_tf):
    """
    Left multiplies a transfer function by a matrix of constants
    :param matrix: A constant matrix
    :param my_tf: A transfer function object from control
    :return: new_tf: A transfer function object from control
    """
    # We build a tf using the matrix as its numerator and
    # an appropriately sized list of lists of arrays of 1 for the denominator.
    dim = np.shape(matrix)
    num = []
    den = []
    for i in range(dim[0]):
        num.append([])
        den.append([])
        for j in range(dim[1]):
            num[i].append(np.array([float(matrix[i, j])]))
            den[i].append(np.array([1.]))
    matrix_tf = tf.tf(num, den)
    # Now we multiply the systems
    new_tf = matrix_tf * my_tf
    # Finally we return our new system
    return new_tf


# Here is a function to preform TM for matrix M and tf T
def left_mult_matrix_by_tf(matrix, my_tf):
    """
    right multiplies a transfer function by a matrix of constants
    :param matrix: A constant matrix
    :param my_tf: A transfer function object from control
    :return: new_tf: A transfer function object from control
    """
    # We build a tf using the matrix as its numerator and
    # an appropriately sized list of lists of arrays of 1 for the denominator.
    dim = np.shape(matrix)
    num = []
    den = []
    flag = True
    for i in range(dim[0]):
        num.append([])
        den.append([])
        for j in range(dim[1]):
            num[i].append(np.array([float(matrix[i, j])]))
            den[i].append(np.array([1.]))
            if num[i][j] != 0:
                flag = False
    if flag:
        num = []
        den = []
        for i in np.arange(np.shape(my_tf.num)[0]):
            num.append([])
            den.append([])
            for j in np.arange(dim[1]):
                num[i].append(np.array([0.]))
                den[i].append(np.array([1.]))
        new_tf = tf.tf(num, den)
        return new_tf
    else:
        matrix_tf = tf.tf(num, den)
        # Now we multiply the systems
        new_tf = my_tf * matrix_tf
        # Finally we return our new system
        return new_tf


# Here is a function to preform T + M for matrix M and tf T
def add_matrix_to_tf(matrix, my_tf):
    """
    matrix: constant matrix
    my_tf: transfer function with same dimension as the input matrix
    :return: new_tf: the resulting tf from the addition
    """
    # We build a tf using the matrix as its numerator and
    # an appropriately sized list of lists of arrays of 1 for the denominator.
    dim = np.shape(matrix)
    num = []
    den = []
    for i in range(dim[0]):
        num.append([])
        den.append([])
        for j in range(dim[1]):
            num[i].append(np.array([float(matrix[i, j])]))
            den[i].append(np.array([1.]))
    matrix_tf = tf.tf(num, den)
    # Now we add the tfs
    new_tf = matrix_tf + my_tf
    return new_tf


# Here is a function to diagnalize a tf
def diagnalize_tf(my_tf):
    """
    Diagnalizes a matrix of transfer functions
    :param my_tf: the input tf object
    :return: diag_tf: the outputed tf object
    """
    num = my_tf.num
    den = my_tf.den
    for i in np.arange(len(num)):
        for j in np.arange(len(num[i])):
            if i != j:
                num[i][j] = np.array([0.])
    diag_tf = tf.tf(num, den)
    return diag_tf


# A function that takes two tf and concatenates them on the desired axis
def tf_concatenate(tf1, tf2, axis):
    """
    takes two transfer functions and concatenates them on the desired axis, tf1 is on top or on the left
    :param tf1: a tf object
    :param tf2: a second tf object
    :param axis: 0 or 1, 0 is vertical and 1 is horizontal concatination
    :return: resulting_tf, a single tf type object
    """
    tf1num = tf1.num
    tf1den = tf1.den
    tf2num = tf2.num
    tf2den = tf2.den
    new_num = np.concatenate((tf1num, tf2num), axis=axis)
    new_den = np.concatenate((tf1den, tf2den), axis=axis)
    # Here we are going to change the arrays into a list of list of arrays to go into the tf builder
    n = np.shape(new_num)
    tfmatrix = []
    denmatrix = []
    # Below we turn the value of our matrix into something that can be made in to a single transfer function so that
    # We can invert it using our inversion function.
    for i in np.arange(n[0]):
        tfmatrix.append([])
        denmatrix.append([])
        for j in np.arange(n[1]):
            tfmatrix[i].append(new_num[i, j])
            denmatrix[i].append(new_den[i, j])
    resulting_tf = tf.tf(tfmatrix, denmatrix)
    return resulting_tf


# A function that builds an identity tf object:
def build_identity_tf(dim):
    """
    makes an identity tf object
    :param dim: the dimension of the desired tf
    :return: identity_tf
    """
    nums = []
    dens = []
    for i in np.arange(dim):
        nums.append([])
        dens.append([])
        for j in np.arange(dim):
            if i == j:
                nums[i].append(np.array([1.]))
                dens[i].append(np.array([1.]))
            else:
                nums[i].append(np.array([0.]))
                dens[i].append(np.array([1.]))
    identity_tf = tf.tf(nums, dens)
    return identity_tf


# A function that computes the Dynamical Structure Function for a given state space model
def find_PQ(A, B, C, D):
    """Will take a LTI with matricies A, B, C, D and then compute the DSF of the system
    the outputs will be matricies of rational functions in terms of the laplace variable.
    This function will check to see that C is of full rank and if it is not it will reorder
    the rows of C so that the top rows are all of full rank."""
    # first we will reorder C such that the linearly independent rows of C are the first
    A, B, C, D, C_1 = order_li_rows(A, B, C, D)
    E_1 = nullspace(C_1)
    # Now we begin the process of finding P and Q
    # We create the nxn state space transformation and call it T
    # To do so we compute a basis of the null space of C_1, the linearly independent rows of C, we call it E_1
    T = np.transpose(np.concatenate((np.transpose(C_1), E_1), axis=1))
    # We calculate T inverse
    T_inv = []
    trow, tcol = np.shape(T)
    for i in np.arange(trow):
        T_inv.append([])
        for j in np.arange(tcol):
            T_inv[i].append(float(T[i, j]))
    T_inv = np.matrix(T_inv)
    T_inv = np.matrix(T_inv)
    T_inv = T_inv.getI()
    # Now we change the basis such that z = Tx, and we solve for z
    #  Doing so will give us: zbar = TAT^-1(z)+TB(u), this gives us some nice results for y.
    A_hat = dot(T, A)
    A_hat = dot(A_hat, T_inv)
    B_hat = dot(T, B)
    C_hat = dot(C, T_inv)
    # D will not be affected by this change of basis
    # We partition our new matrices
    num_known = C_1.shape[0]
    phat = True
    if num_known == np.shape(A)[0]:
        A_part = [A_hat]
        B_part = [B_hat]
        C_part = [C_hat]
        D_part = [D]
        A_chunk = make_matrix_a_tf(np.matrix(A_part[0]))
        B_chunk = make_matrix_a_tf(np.matrix(B_part[0]))
        phat = False
        D_final = A_chunk
        W = A_chunk
        V = B_chunk
    else:
        A_part = partition_matrix(A_hat, num_known, num_known)
        B_part = partition_matrix(B_hat, num_known, B_hat.shape[1])
        C_part = partition_matrix(C_hat, num_known, num_known)
        D_part = partition_matrix(D, num_known, D.shape[1])
        sIminA_22 = simplify_tf(find_resolvent(A_part[3]))  # This is taking lots of time to do...
        A_chunk = right_mult_matrix_by_tf(A_part[1], sIminA_22)
        A_chunk = left_mult_matrix_by_tf(A_part[2], A_chunk)  # Lots of parts of A combined
        B_chunk = right_mult_matrix_by_tf(A_part[1], sIminA_22)
        B_chunk = left_mult_matrix_by_tf(B_part[2], B_chunk)
        D_final = make_matrix_a_tf(D_part[2]) - make_matrix_a_tf(C_part[2]) * make_matrix_a_tf(D_part[0])
        W = add_matrix_to_tf(A_part[0], A_chunk)
        V = add_matrix_to_tf(B_part[0], B_chunk)
    Dw = diagnalize_tf(deepcopy(W))
    # We create a transfer function sI and subtract it from our Dw then take the inverse of the difference
    Dw_num = Dw.num
    sI_num = []
    sI_den = []
    for i in np.arange(len(Dw_num)):
        sI_num.append([])
        sI_den.append([])
        for j in np.arange(len(Dw_num[i])):
            if i == j:
                sI_num[i].append(np.array([1., 0.]))
            else:
                sI_num[i].append(np.array([0.]))
            sI_den[i].append(np.array([1.]))
    sIminDw = tf.tf(sI_num, sI_den) - Dw
    # We now invert this diagnal matrix which means we flip each of the diagnal entries...
    sDw_num, sDw_den = [sIminDw.num, sIminDw.den]
    try:
        leng, wid = np.shape(sDw_num)
    except:
        leng, wid, other = np.shape(sDw_num)
    for l in np.arange(leng):
        for w in np.arange(wid):
                if l != w:
                    sDw_num[l][w], sDw_den[l][w] = [sDw_den[l][w], sDw_num[l][w]]
    sIminDw = tf.tf(sDw_den, sDw_num)
    # We now calculate P and Q
    Q = sIminDw * (W - Dw)
    P = sIminDw * V
    if np.shape(D_final.num)[0] != 0:
        P_up = left_mult_matrix_by_tf(D_part[0], build_identity_tf(np.shape(Q.num)[0]) - Q)
        P_up = P + P_up
        if phat:
            P_down = D_final
            if np.any(P_down.num) != 0:
                P_hat = tf_concatenate(P_up, P_down, 0)
            else:
                P_hat = P_up
        else:
            P_hat = P_up
    else:
        P_hat = P + left_mult_matrix_by_tf(D_part[0], build_identity_tf(np.shape(Q.num)[0]) - Q)
    if phat:
        if np.shape(C_part[2])[0] != 0:
            Q_up = Q
            Q_down = make_matrix_a_tf(C_part[2])
            Q_hat = tf_concatenate(Q_up, Q_down, 0)
        else:
            Q_hat = Q
    else:
        Q_hat = Q
    return [P_hat, Q_hat]


# A function to simplify P and Q if they are huge
def simplify_tf(my_tf_object):
    """
    :param: my_tf is a transfer function object
    :return: simp_tf
    """
    numArray_reverse = my_tf_object.num
    denArray_reverse = my_tf_object.den
    # Now we take the coefficients of the polynomial and put them into 2 arrays
    # one for numerators and one for denominators
    numArray_list = list(numArray_reverse)
    denArray_list = list(denArray_reverse)
    try:
        row, col = np.shape(denArray_list)
    except:
        row, col, other = np.shape(denArray_list)
    for i in np.arange(row):
        for j in np.arange(col):
            numArray_list[i][j] = np.flipud(numArray_list[i][j])
            denArray_list[i][j] = np.flipud(denArray_list[i][j])
    numArray_pres = (numArray_list)
    denArray = (denArray_list)
    numArray_fut = deepcopy(numArray_pres)

    denArray_fut = deepcopy(denArray)
    # Below we do the denominator:
    # Now we collect our results and we return our final inverted matrix
    my_adjoint_num = deepcopy(numArray_fut)
    my_adjoint_den = deepcopy(denArray_fut)
    # We multiply the adjoint by the inverse of the determinant of the matrix, the inverse of the determinant is
    # decided to be the past numerator array (or that before preforming the final pivot)
    inverse_determinant_den = [1]
    inverse_determinant_num = [1]
    newMatrix_num = []
    newMatrix_den = []
    for i in np.arange(row):
        newMatrix_num.append([])
        for j in np.arange(col):
            newMatrix_num[i].append(tuple(poly.polymul(my_adjoint_num[i][j], inverse_determinant_num)))
    # Now we do the denominators
    for i in np.arange(row):
        newMatrix_den.append([])
        for j in np.arange(col):
            x = np.shape(my_adjoint_den)
            newMatrix_den[i].append(tuple(poly.polymul(my_adjoint_den[i][j], inverse_determinant_den)))
    # Ok, we finally have our inverse!
    numerators = newMatrix_num
    denominators = newMatrix_den
    try:  # A polynomial solver for smaller degree polynomials to do cancellations and reduce the polynomials
        for i in np.arange(row):
            for j in np.arange(col):
                numerators[i][j] = np.flipud(numerators[i][j])
                denominators[i][j] = np.flipud(denominators[i][j])
                k = 0
                for val in numerators[i][j]:
                    numerators[i][j][k] = round(val, 4)
                    k += 1
                num_roots = (np.roots(numerators[i][j]))
                den_roots = (np.roots(denominators[i][j]))
                for r in np.arange(len(num_roots)):
                    num_roots[r] = round(num_roots[r], 4) * 10000.
                for dr in np.arange(len(den_roots)):
                    den_roots[dr] = round(den_roots[dr], 4) * 10000.
                num_roots = Counter(num_roots)
                den_roots = Counter(den_roots)
                all_roots_num = num_roots - den_roots
                all_roots_den = den_roots - num_roots
                if np.any(numerators[i][j]) != 0:  # If zero then we want to keep them as they are.
                    numerators[i][j] = numerators[i][j][0] / denominators[i][j][0]
                for k in all_roots_num:
                    for number in np.arange(all_roots_num[k]):
                        numerators[i][j] = list(poly.polymul(numerators[i][j], (-k / 10000., 1)))
                denominators[i][j] = 1
                for k in all_roots_den:
                    for number in np.arange(all_roots_den[k]):
                        denominators[i][j] = list(poly.polymul(denominators[i][j], (-k / 10000., 1)))
        # Now we return the values in the form of transfer functions by casting them with control.tf
        for i in np.arange(row):
            for j in np.arange(col):
                if type(numerators[i][j]) == 'npumpy.ndarray' or type(numerators[i][j]) == list:
                    numerators[i][j] = np.flipud(numerators[i][j])
                else:
                    numerators[i][j] = np.array([float(np.real(numerators[i][j]))])
                if type(denominators[i][j]) != int:
                    denominators[i][j] = np.flipud(denominators[i][j])
                else:
                    denominators[i][j] = np.array([float(denominators[i][j])])
    except:
        b = "c"
    num_list = []
    den_list = []
    for i in np.arange(row):
        num_list.append(list(numerators[i]))
        den_list.append(list(denominators[i]))
    simp_tf = tf.tf(num_list, den_list)
    return simp_tf


# A function that modifies C so that only selected states are viewable
def build_C(A, B, C, D, indicies):
    """ This function rewrites C and D in the system to correspond to the state variables that were chosen as hidden
    or manifest.
    :param:
    A, B, C, D are state space matricies for an LTI, here D and C should be such that we assume that all the state
    variables are manifest.
    indicies, a 1-D array of integers, the states of the system that are not hidden
    :return:
    A, B, new_C, new_D: matricies of real numbers corresponding the system
    """
    # First we build our new C
    num_rows = len(indicies)
    new_C = np.zeros([num_rows, np.shape(A)[1]])
    for i in np.arange(num_rows):
        new_C[i, indicies[i]] = 1
    # Now we build our new D
    new_D = np.zeros(np.shape(B)[1])
    for index in indicies:
        new_D = np.vstack([new_D, D[index, :]])
    return A, B, new_C, new_D[1:, :]


# Here is a function for choosing our initial starting point
def initial_gamma(my_tf):
    """
    Finds the initial gamma to use for the algorithm
    :param my_tf:
    :return: gamma: a singular value to use as our base case
    """
    # We ultimately want to choose the greatest singular value of three, the first two are straightforward:
    sig_0 = abs(my_tf(0))
    sig_inf = abs(my_tf(350))
    # Now we want to choose a third value that is j*(our favorite pole)
    try:
        poles = tf.pole(my_tf)
        w = -np.inf
        if np.any(np.imag(poles)) != 0:
            for lamb in poles:
                lamby = abs((np.imag(lamb) / np.real(lamb)) * (1. / abs(lamb)))
                if lamby > w:
                    w = abs(lamb)
            sig_w = abs(my_tf(w * 1.0j))
        else:
            if np.all(np.real(poles)) == 0:  # Case where all poles are 0
                sig_w = abs(my_tf(0))
            else:
                sig_w = np.min(abs(poles))
                sig_w = abs(my_tf(sig_w * 1.0j))
    except AssertionError:
        sig_w = -np.inf
    gamma = np.max([sig_0, sig_w, sig_inf])
    return gamma


# here is a function for computing H(gamma)
def build_H(A, B, C, D, gamma):
    """
    Builds the Hamiltoniam matrix, whose eigenvalues we use to discover the H_inf matrix.
    :param A: matrix
    :param B: matrix
    :param C: matrix
    :param D: 1 by 1 matrix
    :param gamma: float (maybe complex)
    :return: H: a matrix
    """
    R_S = 1./(D**2 - gamma**2)
    ul = A - B * R_S * D * C
    ur = -gamma * B * R_S * B.T
    ll = gamma * C.T * R_S * C
    lr = -A.T + C.T * D * R_S * B.T
    upper = np.hstack([ul, ur])
    lower = np.hstack([ll, lr])
    H = np.vstack([upper, lower])
    return H


# Here is the function that computes the H-infinity norm of a SISO tf
def siso_h_inf(my_tf, error, max_iters=120):
    """
    Calculates the H-infinity norm of a transfer function for a SISO system.
    Uses the algorithm from: "A fast algorithm to compute the H -norm of a transfer function matrix"
    By: N.A. BRUINSMA and M. STEINBUCH, by the way, this is the algorithm that matlab cites for their norm function.
    :param my_tf: A SISO transfer function object
           error: a float representing the level of accuracy we want to approximate the infinity norm to
           max_iters: int, the max number of iterations that we are willing to do
    :return: h_inf_norm : a float or numpy.inf
    """
    # First we find the A, B, C, D matricies associated with my_tf and extract them as matrix objects
    if np.sum(np.abs(my_tf.num)) == 0:  # If the transfer function is zero, the norm is 0
        return 0.0
    if my_tf(1.234) == my_tf(524.653):  # Make sure that it is not a zero transfer function.
        return 0.0
    system = tf.tf2ss(my_tf)
    A = system.A
    B = system.B
    C = system.C
    D = system.D
    # Check if A has imaginary eigenvalues, if so return np.inf
    A_eigs = np.linalg.eig(A)[0]
    for eig in A_eigs:
        if np.real(eig) == 0:
            if np.imag(eig) != 0:
                h_inf_norm = np.inf
                return h_inf_norm
    # Step I, choose a good starting lower bound (starting point)
    gamma = initial_gamma(my_tf)
    gamma1 = deepcopy(gamma)
    # begin a for loop
    for iter in np.arange(max_iters):
        if gamma == (np.linalg.eig(D)[0][0])**0.5:  # check that gamma is not a singular value of D
            gamma *= (1 - 0.5 * error)
        H = build_H(A, B, C, D, (gamma*(1 + 2*error)))
        # Step II, check the stopping criteria
        flag = True
        try:
            H_eigs = np.linalg.eig(H)[0]
        except np.linalg.linalg.LinAlgError:
            return max(gamma, gamma1)
        for eig in H_eigs:
            if np.iscomplex(eig):  # If the stopping criteria is met break out of the loop
                if abs(np.real(eig)) < 10**-10:  # Approximation
                    flag = False
        if flag:
            break
        # Step III Find our new starting point
        ws = []
        ms = []
        sing_m = []
        for eig in H_eigs:
            if np.iscomplex(eig):  # If the stopping criteria is met break out of the loop
                if abs(np.real(eig)) < 10**-10:
                    ws.append(np.imag(eig))
        ws.sort()
        for i in np.arange(len(ws) - 1):
            ms.append((ws[i] + ws[i+1]) * 0.5)
        for m in ms:
            sing_m.append(abs(my_tf(m*1.0j)))
        gamma = np.max(sing_m)
    # The case where we finished our max iterations, or we have met the stopping criteria
    # output the best estimate made in our iterations (the most recent)
    # set ||G||inf = gamma(1+error)
    h_inf_norm = gamma * (1 + error)
    return h_inf_norm


# Here is a function that takes a transfer function and returns a matrix of zeros and ones, where zeros are at each entry of the transfer function equal to zero.
def bool_tf(my_tf):
    """
    takes a transfer function and returns a matrix of zeros and ones
    :param my_tf: a transfer function object
    :return: bool_matrix, a 2D array of zeros and ones
    """
    num = my_tf.num
    r, c = np.shape(num)[0:2]
    bool_tf = np.ones([r,c])
    for i in np.arange(r):
        for j in np.arange(c):
            if len(num[i][j]) == 1:
                if num[i][j] == 0:
                    bool_tf[i, j] = 0
    return bool_tf


# Here is a function that scores each link for vulnerability and returns the weaknesses in a matrix
def vuln_Q(Q, error, iters=100):
    """
    :param:
    Q a transfer function
    error: real number for how accurate you want your vulnerability scores to be
    iters: number of max attempts we are willing to make in finding vulnerability of any given link
    :return:
    scores: a matrix of floats showing the vulnerability from link i to j in each entry
    """
    rows = np.shape(Q.num)[0]
    # Case where there is only one exposed variable:
    if Q.num == [[np.array([0.])]]:
        return [[0]]
    # H = (I-Q)^-1
    i_min_Q = -add_matrix_to_tf(-np.eye(rows), Q)
    H = inverse_rational_matrix(i_min_Q)
    if len(list(np.shape(H.num))) == 3:
        Hrows, Hcols, Shell = np.shape(H.num)
    else:
        Hrows, Hcols = np.shape(H.num)
    forescores = np.zeros([Hrows, Hcols])
    scores = np.zeros([Hrows, Hcols])
    # Now we determine the elements of Q that are in some cycle and therefore have some vulnerability
    Qbool = bool_tf(Q) + np.eye(rows)
    Qboolorig = deepcopy(Qbool)
    for i in np.arange(Hrows):
        Qbool = dot(Qbool, Qboolorig)
    no_cycle_nodes = []
    for i in np.arange(rows):
        for j in np.arange(rows):
            if i == j:
                if Qbool[i,j] == 1:
                    no_cycle_nodes.append(i)
    for i in np.arange(Hrows):
        for j in np.arange(Hcols):
            if i != j:
                Hnum = np.real(H.num[i][j])
                if np.sum(np.abs(Hnum)) != 0 and np.sum(np.abs(Q.num[j][i])) != 0:
                    if i not in no_cycle_nodes and j not in no_cycle_nodes:
                        Hij = tf.tf(Hnum, np.real(H.den[i][j]))
                        forescores[i][j] = (siso_h_inf(Hij, error, iters))
            else:
                forescores[i][j] = 0
    for i in np.arange(Hrows):  # I believe this is how the vulnerability is assigned...
        for j in np.arange(Hcols):
            scores[j][i] = forescores[i][j]
    return scores


# Here is a function to give our output the correct format for visualization
def format_pq(my_matrix):
    """
    returns my_matrix as a list of lists and puts a cap on each entry
    :return: out: a list of lists
    """
    row, col = np.shape(my_matrix)
    out = []
    for i in np.arange(row):
        out.append([])
        for j in np.arange(col):
            if my_matrix[i, j] < 350:
                out[i].append(my_matrix[i, j])
            else:
                out[i].append(350)
    return out



def ret_PQ(A, B, C, D, index):
    """
    Returns P and Q ready for visualization and with vulnerability scores as well.
    :param A: matrix
    :param B: matrix
    :param C: matrix
    :param D: matrix
    :param index: list of incidices of exposed variables
    :return: P and Q
    """
    A = np.matrix(A)
    B = np.matrix(B)
    C = np.matrix(C)
    D = np.matrix(D)
    print ("A.shape",A.shape)
    print ("B.shape",B.shape)
    print ("C.shape",C.shape)
    print ("D.shape",D.shape)
    A,B,C,D = build_C(A, B, C, D, index)
    P, Q = find_PQ(A, B, C, D)
    P = simplify_tf(P)
    Q = simplify_tf(Q)
    vuln = vuln_Q(Q, 0.01)
    Pout = format_pq(bool_tf(P))
    Qout = format_pq(bool_tf(Q)+vuln)
    return Pout, Qout
