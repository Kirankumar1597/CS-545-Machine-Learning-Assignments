run_my_solution = False

import os
import copy
import signal
import os
import numpy as np

if run_my_solution:
    from A2mysolution import *
    # print('##############################################')
    # print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    # print('##############################################')

else:
    
    print('\n======================= Code Execution =======================\n')

    assignmentNumber = '2'

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

for func in ['Optimizers', 'NeuralNetwork', 'partition', 'run_experiment']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')

            
print('''\nTesting
  w = np.array([0.0])
  def cubic(wmin):
      return (w[0] - wmin) ** 3 + (w[0] - wmin) ** 2
  def grad_cubic(wmin):
      return 3 * (w[0] - wmin) ** 2 + 2 * (w[0] - wmin)
  wmin = 0.5
  opt = Optimizers(w)
  errors_sgd = opt.sgd(cubic, grad_cubic, [wmin], 100, 0.01)
''')


try:
    pts = 10

    w = np.array([0.0])
    def cubic(wmin):
        return (w[0] - wmin) ** 3 + (w[0] - wmin) ** 2
    def grad_cubic(wmin):
        return 3 * (w[0] - wmin) ** 2 + 2 * (w[0] - wmin)
    wmin = 0.5
    opt = Optimizers(w)
    errors_sgd = opt.sgd(cubic, grad_cubic, [wmin], 100, 0.01)
    
    correct_sgd = 0.030721
    
    if np.allclose(errors_sgd[-1], correct_sgd, 0.05):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct value.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. Final error should be')
        print(correct_sgd)
        print(f'        Your value is')
        print(errors_sgd[-1])
except Exception as ex:
    print(f'\n--- 0/{pts} points. Optimizers.sgd raised the exception\n')
    print(ex)


print('''\nTesting
  w = np.array([0.0])
  def cubic(wmin):
      return (w[0] - wmin) ** 3 + (w[0] - wmin) ** 2
  def grad_cubic(wmin):
      return 3 * (w[0] - wmin) ** 2 + 2 * (w[0] - wmin)
  wmin = 0.5
  opt = Optimizers(w)
  errors_adam = opt.adam(cubic, grad_cubic, [wmin], 100, 0.01)
''')


try:
    pts = 10

    w = np.array([0.0])

    def cubic(wmin):
        return (w[0] - wmin) ** 3 + (w[0] - wmin) ** 2
    
    def grad_cubic(wmin):
        return 3 * (w[0] - wmin) ** 2 + 2 * (w[0] - wmin)
    
    wmin = 0.5
    
    w = np.array([0.0])
    opt = Optimizers(w)
    errors_adam = opt.adam(cubic, grad_cubic, [wmin], 100, 0.01)

    correct_adam =  8.541370261709512e-06
    
    if np.allclose(errors_adam[-1], correct_adam, 0.1):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct value.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. Final error should be')
        print(correct_adam)
        print(f'        Your value is')
        print(errors_adam[-1])
except Exception as ex:
    print(f'\n--- 0/{pts} points. Optimizers.adam raised the exception\n')
    print(ex)

    

print('''\nTesting
    np.random.seed(42)
    
    nnet = NeuralNetwork(2, [10], 1)
    X = np.arange(40).reshape(20, 2)
    T = X[:, 0:1] * X[:, 1:]
    nnet.train(X, T, 1000, 0.01, method='adam')
''')

try:
    pts = 20

    np.random.seed(42)
    
    nnet = NeuralNetwork(2, [10], 1)
    X = np.arange(40).reshape(20, 2)
    T = X[:, 0:1] * X[:, 1:]
    nnet.train(X, T, 1000, 0.01, method='adam')
    
    answer= 5.089412379846469

    if np.allclose(nnet.error_trace[-1], answer, 0.2):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct value.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. Final error should be')
        print(answer)
        print(f'        Your value is')
        print(nnet.error_trace[-1])
except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetwork constructor or train raised the exception\n')
    print(ex)


print('''\nTesting
    np.random.seed(42)
    
    # Using X and T from previous test
    a, b, c, d, e, f = partition(X, T, 3)
''')


try:
    pts = 10

    np.random.seed(42)
    
    # Using X and T from previous test
    a, b, c, d, e, f = partition(X, T, 3)

    correct_e = np.array([[22, 23],
                          [ 6,  7],
                          [36, 37],
                          [32, 33],
                          [26, 27],
                          [ 4,  5]])
    correct_f = np.array([[ 506],
                          [  42],
                          [1332],
                          [1056],
                          [ 702],
                          [  20]])

    if (e.shape == (6, 2) and f.shape == (6, 1) and
        np.allclose(e, correct_e, 0.01) and
        np.allclose(f, correct_f, 0.01)):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. e and f should be')
        print(correct_e)
        print(correct_f)
        print(f'        Your values are')
        print(e)
        print(f)
except Exception as ex:
    print(f'\n--- 0/{pts} points. partition raised the exception: {ex}\n')
    # print(f'\n or returned incorrect values. e and f should be')
    # print(correct_e)
    # print(correct_f)
    # print(f'        Your values are')
    # print(e)
    # print(f)

    

print('''\nTesting
    np.random.seed(42)

    result = run_experiment(X, T, 3,
                            n_epochs_choices=[10, 20],
                            n_hidden_units_per_layer_choices=[[0], [10]],
                            activation_function_choices=['tanh', 'relu'])

    first_test_rmse = result.iloc[0]['RMSE Test']
''')

try:
    pts = 20

    np.random.seed(42)

    result = run_experiment(X, T, 3,
                            n_epochs_choices=[10, 20],
                            n_hidden_units_per_layer_choices=[[0], [10]],
                            activation_function_choices=['tanh', 'relu'])

    first_test_rmse = result.iloc[0]['RMSE Test']
        
    answer = 150.907

    if np.allclose(first_test_rmse, answer, 0.1):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. first_test_rmse should be')
        print(answer)
        print(f'        Your value is')
        print(first_test_rmse)
except Exception as ex:
    print(f'\n--- 0/{pts} points. run_experiment raised the exception\n')
    print(ex)

    



name = os.getcwd().split('/')[-1]

print()
print('='*70)
print('{} Execution Grade is {} / 70'.format(name, exec_grade))
print('='*70)


print('''\n __ / 30 Discussion of at least three observations about
your results.  Please be detailed enough that your conclusions are clear.''')

print()
print('='*70)
print('{} FINAL GRADE is  _  / 100'.format(name))
print('='*70)

print('''
Extra Credit:
Add the Swish activation function as a third choice in your train function in your NeuralNetwork class. A little googling will find definitions of it and its gradient.

Use your run_experiment function to compare results for all three activation functions. Discuss the results.''')

print('\n{} EXTRA CREDIT is 0 / 1'.format(name))

if run_my_solution:
    # print('##############################################')
    # print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    # print('##############################################')
    pass

