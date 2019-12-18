import glob
import os
import shelve
import subprocess as shell
import sys
from datetime import datetime

def dateNow(format = None):
    if(format != None):
        return(datetime.now().strftime(format))
    return(str(datetime.now()))

def delete(directory, pattern):
    # Get a list of all the file paths that math pattern inside directory
    fileList = glob.glob(directory + '/' + pattern)

    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print('Error while deleting file : ' + filePath)
    # Reference: https://thispointer.com/python-how-to-remove-files-by-matching-pattern-wildcards-certain-extensions-only/

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[0])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)

def load(filename):
    existingShelf = shelve.open(filename)
    for key in existingShelf:
        os.environ[key]=existingShelf[key]
    existingShelf.close()
    # Reference: https://stackoverflow.com/questions/2960864/how-can-i-save-all-the-variables-in-the-current-python-session

def progbar(curr, total, full_progbar = 40):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')
    sys.stdout.flush()

def save(filename, variables):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory) and directory != '':
        os.makedirs(directory)
    newShelf = shelve.open(filename,'n') # 'n' for new
    if(type(variables) != type(list())):
        variables = [variables]
    for key in variables:
        try:
            newShelf[key] = os.environ[key]
        except TypeError:
            #
            # __builtins__, newShelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    newShelf.close()
    # Reference: https://stackoverflow.com/questions/2960864/how-can-i-save-all-the-variables-in-the-current-python-session

def sbatch(jobFile, dependency):
    if(dependency == ''):
        proc = shell.Popen(['sbatch ' + jobFile], shell = True, stdout = shell.PIPE)
    else:
        proc = shell.Popen(['sbatch --dependency=afterok:' + dependency + ' ' + jobFile], shell = True, stdout = shell.PIPE)
    result = str(proc.communicate()[0].decode('ascii').strip())
    print(result)
    return(result.split(' ')[-1])
