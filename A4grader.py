run_my_solution = False

import os
import copy
import signal
import os
import numpy as np

if run_my_solution:
    from A4mysolution import *
    # print('##############################################')
    # print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    # print('##############################################')

else:
    
    print('\n======================= Code Execution =======================\n')

    assignmentNumber = '4'

    import subprocess, glob, pathlib
    nb_name = '*A{}*.ipynb'
    # nb_name = '*.ipynb'
    filename = next(glob.iglob(nb_name.format(assignmentNumber)), None)
    print('Extracting python code from notebook named \'{}\' and storing in notebookcode.py'.format(filename))
    if not filename:
        raise Exception('Please rename your notebook file to <Your Last Name>-A{}.ipynb'.format(assignmentNumber))
    with open('notebookcode.py', 'w') as outputFile:
        subprocess.call(['jupyter', 'nbconvert', '--to', 'script',
                         nb_name.format(assignmentNumber), '--stdout'], stdout=outputFile)
    # from https://stackoverflow.com/questions/30133278/import-only-functions-from-a-python-file
    import sys
    import ast
    import types
    with open('notebookcode.py') as fp:
        tree = ast.parse(fp.read(), 'eval')
    print('Removing all statements that are not function or class defs or import statements.')
    for node in tree.body[:]:
        if (not isinstance(node, ast.FunctionDef) and
            not isinstance(node, ast.Import) and
            not isinstance(node, ast.ClassDef)):
            # not isinstance(node, ast.ImportFrom)):
            tree.body.remove(node)
    # Now write remaining code to py file and import it
    module = types.ModuleType('notebookcodeStripped')
    code = compile(tree, 'notebookcodeStripped.py', 'exec')
    sys.modules['notebookcodeStripped'] = module
    exec(code, module.__dict__)
    # import notebookcodeStripped as useThisCode
    from notebookcodeStripped import *


    
exec_grade = 0

for func in ['NeuralNetwork', 'NeuralNetworkClassifier']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')

            
print('''\nTesting

    nn_reg = NeuralNetwork(1, [5], 2)
    nn_class = NeuralNetworkClassifier(1, [5], 5)

    nn_reg.forward_pass.__func__ == nn_class.forward_pass.__func__
''')


try:
    pts = 5

    nn_reg = NeuralNetwork(1, [5], 2)
    nn_class = NeuralNetworkClassifier(1, [5], 5)

    if nn_reg.forward_pass.__func__ == nn_class.forward_pass.__func__:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points.  Function NeuralNetwork.forward_pass is correctly not overridden in NeuralNetworkClassifier.')
    else:
        print(f'\n---  0/{pts} points. Function NeuralNetwork.forward_pass should not be overridden in NeuralNetworkClassifier.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. Raises exception:')
    print(ex)
        


print('''\nTesting

    nn_reg = NeuralNetwork(1, [5], 2)
    nn_class = NeuralNetworkClassifier(1, [5], 5)

    nn_reg.train.__func__ != nn_class.train.__func__
''')


try:
    pts = 5

    nn_reg = NeuralNetwork(1, [5], 2)
    nn_class = NeuralNetworkClassifier(1, [5], 5)

    if nn_reg.train.__func__ != nn_class.train.__func__:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points.  Function NeuralNetwork.train is correctly overridden in NeuralNetworkClassifier.')
    else:
        print(f'\n---  0/{pts} points. Function NeuralNetwork.train should be overridden in NeuralNetworkClassifier.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. Calls to constructors the exception:')
    print(ex)
        



print('''\nTesting

    nn_reg = NeuralNetwork(1, [5], 2)
    nn_class = NeuralNetworkClassifier(1, [5], 5)

    nn_reg.use.__func__ != nn_class.use.__func__
''')


try:
    pts = 5

    nn_reg = NeuralNetwork(1, [5], 2)
    nn_class = NeuralNetworkClassifier(1, [5], 5)

    if nn_reg.use.__func__ != nn_class.use.__func__:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points.  Function NeuralNetwork.use is correctly overridden in NeuralNetworkClassifier.')
    else:
        print(f'\n---  0/{pts} points. Function NeuralNetwork.use should be  overridden in NeuralNetworkClassifier.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. Exception raised:')
    print(ex)

        
print('''\nTesting

    nn_class = NeuralNetworkClassifier(1, [5], 5)
    result = nn_class.makeIndicatorVars(np.arange(5).reshape(-1, 1))

''')

try:
    pts = 5

    nn_class = NeuralNetworkClassifier(1, [5], 5)
    result = nn_class.makeIndicatorVars(np.arange(5).reshape(-1, 1))
    answer = np.eye(5)
    
    if np.allclose(nn_class.makeIndicatorVars(np.arange(5).reshape(-1, 1)), answer):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points.  makeIndicatorVars correctly returned all 1''s on diagonal.')
    else:
        print(f'\n---  0/{pts} points. makeIndicatorVars should return\n')
        print(answer)
        print(' but you returned')
        print(result)
except Exception as ex:
    print(f'\n--- 0/{pts} points. Exception raised:')
    print(ex)
        

print('''\nTesting

    nn_class = NeuralNetworkClassifier(1, [5], 5)
    result = nn_class.softmax(np.array([[-5.5, 5.5]]))

''')


try:
    pts = 10

    nn_class = NeuralNetworkClassifier(1, [5], 5)
    result = nn_class.softmax(np.array([[-5.5, 5.5]]))
    answer = np.array([[1.67014218e-05, 9.99983299e-01]])
    
    if np.allclose(result, answer, 0.1):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points.  softmax returned correct answer.')
    else:
        print(f'\n---  0/{pts} points. softmax should return\n')
        print(answer)
        print(' but you returned')
        print(result)
except Exception as ex:
    print(f'\n--- 0/{pts} points. Exception raised:')
    print(ex)
        





print('''\nTesting

    X = np.arange(20).reshape(20, 1)
    X = np.hstack((X, X[::-1, :]))
    T = np.array(['ends', 'mid'])[(np.abs(X[:, 0:1] - X[:, 1:2]) < 6).astype(int)]

    np.random.seed(42)
    
    nnet = NeuralNetworkClassifier(X.shape[1], [10, 10], len(np.unique(T)), activation_function='relu')
    nnet.train(X, T, 500, 0.001, method='adam', verbose=False)

    Y_classes, Y_probs = nnet.use(X)

    percent_correct = 100 * np.mean(Y_classes == T)
''')

try:
    pts = 10
    
    X = np.arange(20).reshape(20, 1)
    X = np.hstack((X, X[::-1, :]))
    T = np.array(['ends', 'mid'])[(np.abs(X[:, 0:1] - X[:, 1:2]) < 6).astype(int)]

    np.random.seed(42)
    
    nnet = NeuralNetworkClassifier(X.shape[1], [10, 10], len(np.unique(T)), activation_function='relu')
    nnet.train(X, T, 500, 0.001, method='adam', verbose=False)

    Y_classes, Y_probs = nnet.use(X)

    percent_correct = 100 * np.mean(Y_classes == T)
    
    answer = 100.0

    if np.allclose(percent_correct, answer, 0.1):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Correctly returned {answer}.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect value. percent_correct should be {answer}. Your value is {percent_correct}.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. Constructor or train raised the exception:')
    print(ex)



print('''\nTesting

    cm = confusion_matrix(Y_classes, T)
''')

try:
    pts = 10
    
    cm = confusion_matrix(Y_classes, T)

    array = cm.values
    values = np.array([[100., 0.], [0., 100.]])
    cols = cm.columns.values
    rows = cm.index.values

    class_names = np.unique(T)
    answer = pandas.DataFrame(values, columns=class_names, index=class_names)

    if np.all(values == array) and np.all(cols == class_names) and np.all(rows == class_names):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Correctly returned DataFrame:')
        print(cm)
    else:
        print(f'\n---  0/{pts} points. Did not return correct DataFrame.')
        print(' Correct answer is')
        print(answer)
        print(' You returned')
        print(cm)
except Exception as ex:
    print(f'\n--- 0/{pts} points. confusion_matrix raised the exception:')
    print(ex)

    
    



name = os.getcwd().split('/')[-1]

print()
print('='*70)
print('{} Execution Grade is {} / 50'.format(name, exec_grade))
print('='*70)


print('''\n __ / 50 Based on other testing and the results you obtain and your discussions.''')

print()
print('='*70)
print('{} FINAL GRADE is  _  / 100'.format(name))
print('='*70)

print('''
Extra Credit:

Earn 5 extra credit points on this assignment by doing the following steps.

1. Combine the train, validate, and test partitions loaded from the MNIST data file into 
   two matrices, X and T.

2. Using adam, relu and just one value of learning_rate and n_epochs, compare several 
   hidden layer architectures. Do so by applying our generate_k_fold_cross_validation_sets
   function as defined in Lecture Notes 12 which forms stratified partitioning, for use
   in classification problems, to your X and T matrices using n_fold of 3.

3. Show results and discuss which architectures you find works the best, and how you
    determined this.''')

if run_my_solution:
    # print('##############################################')
    # print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    # print('##############################################')
    pass

