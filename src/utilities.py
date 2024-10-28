# Libraries to load
import numpy as np
import os
from scipy.interpolate import interp1d

# some usefull functions

def writefiles(path,x,y):
    with open(path, 'w') as txtfile:
        for i in range(0,len(x)):
            txtfile.write("%s \t %s\n" %(x[i],y[i]))
        txtfile.close()
    
        
#function that reads a table in a .txt file and converts it to a numpy array
def readfile(filename):
    array = []
    with open(filename) as f:
        for line in f:
            if line[0]=="#":continue
            words = [float(elt.strip()) for elt in line.split( )]
            array.append(words)
    return np.array(array)

def arrdiv(arrA,arrB):
    div = []
    for i in range(0,len(arrA)):
        if arrB[i]==0:
            div.append(0)
        elif arrB[i]!=0:
            div.append(arrA[i]/arrB[i])
    return div

def intdiv(n,d):
    return n/d if d else 0
   
    
# auxiliary functions

def is_all_zero(array):
    return np.all(array == 0)

def sci_not(number):
    exponent = 0
    if number != 0:
        exponent = int(np.floor(np.log10(np.abs(number))))
    mantissa = number / (10 ** exponent)
    return ("${%.2f} \\times 10^{%d} $"  %(mantissa, exponent)) 
    
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# function that transforms arrays in a dictionary into interpolated functions
def interp_dict(input_dict, x_array):
    """
    Transform arrays in a dictionary into interpolated functions.
    
    Parameters:
    - input_dict: A dictionary where keys are names and values are arrays.
    - x_array: A numpy array corresponding to the x values of the interpolated functions.
    
    Returns:
    - int_dict: A dictionary where keys are names and values are interpolated functions.
    """
    int_dict = {}
    
    for name, y_arr in input_dict.items():
        int_dict[name] = interp1d(x_array, y_arr, kind='linear', fill_value="extrapolate")
    
    return int_dict
  
# function that receives a dictionary where the values are interpolated functions fint
# and an array A, and save the fint(A) into the specified filename

def savefiles(func_dict, array, filename):
    """
    Saves interpolated values of functions applied to an array to a text file.

    Parameters:
    func_dict (dict): Dictionary where keys are function names and values are interpolating functions.
    array (np.ndarray): Array of values to be interpolated.
    filename (str): Name of the output text file.
    """
    with open(filename, 'w') as file:
        for name, func in func_dict.items():
            interpolated_values = func(array)
            file.write(f"{name}:\n")
            np.savetxt(file, interpolated_values, newline=" ")
            file.write("\n\n")
