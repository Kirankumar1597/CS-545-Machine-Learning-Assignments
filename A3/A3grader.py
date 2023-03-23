run_my_solution = False

import os
import copy
import signal
import os
import numpy as np

if run_my_solution:
    from A3mysolution import *
    # print('##############################################')
    # print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    # print('##############################################')

else:
    
    print('\n======================= Code Execution =======================\n')

    assignmentNumber = '3'

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

for func in ['NNet', 'run_k_fold_cross_validation']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')

            
print('''\nTesting
    torch.manual_seed(142)

    X = np.arange(40).reshape(20, 2)
    T = np.log(1 + X[:, 0:1])

    nnet = NNet(X.shape[1], [10, 10], T.shape[1], act_func='relu')
    nnet.train(X, T, 100, 0.01, verbose=False)

    first_Y = nnet.use(X)[0, 0]
''')


try:
    pts = 30

    torch.manual_seed(142)
    
    X = np.arange(40).reshape(20, 2)
    T = np.log(1 + X[:, 0:1])

    nnet = NNet(X.shape[1], [10, 10], T.shape[1], act_func='relu')
    nnet.train(X, T, 100, 0.01, verbose=False)

    first_Y = nnet.use(X)[0, 0]

    answer = 0.65859365

    if np.allclose(first_Y, answer, 0.1):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct value.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect value. first_Y should be')
        print(answer)
        print(f'        Your value is')
        print(first_Y)
except Exception as ex:
    print(f'\n--- 0/{pts} points. NNet constructor, train, or use raised the exception\n')
    print(ex)




print('''\nTesting
    torch.manual_seed(142)

    X = np.arange(40).reshape(20, 2)
    T = np.log(1 + X[:, 0:1])

    nnet = NNet(X.shape[1], [10, 10], T.shape[1], act_func='tanh')  # Using tanh
    nnet.train(X, T, 100, 0.01, verbose=False)

    first_Y = nnet.use(X)[0, 0]
''')


try:
    pts = 30

    torch.manual_seed(142)

    X = np.arange(40).reshape(20, 2)
    T = np.log(1 + X[:, 0:1])
    
    nnet = NNet(X.shape[1], [10, 10], T.shape[1], act_func='tanh')  # Using tanh
    nnet.train(X, T, 100, 0.01, verbose=False)

    first_Y = nnet.use(X)[0, 0]

    answer = 0.52785134

    if np.allclose(first_Y, answer, 0.1):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct value.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect value. first_Y should be')
        print(answer)
        print(f'        Your value is')
        print(first_Y)
except Exception as ex:
    print(f'\n--- 0/{pts} points. NNet constructor, train, or use raised the exception\n')
    print(ex)


    




print('''\nTesting
    torch.manual_seed(400)
    np.random.seed(400)

    X = np.arange(40).reshape(20, 2)
    T = np.log(1 + X[:, 0:1])
    
    results = run_k_fold_cross_validation(X, T, 3,
                                          [[], [1], [20], [40, 40]],
                                          1000, 0.01, 'tanh')

    print(results)

    # This will take a minute or two....

''')


try:
    pts = 30

    torch.manual_seed(400)
    np.random.seed(400)

    X = np.arange(40).reshape(20, 2)
    T = np.log(1 + X[:, 0:1])
    
    results = run_k_fold_cross_validation(X, T, 3,
                                          [[], [1], [20], [40, 40]],
                                          1000, 0.01, 'tanh')

    print(results)

    last_train = results.iloc[-1, 1]
    answer = 0.001703
    
    if np.allclose(last_train, answer, 2):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct value.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. Last Train RMSE should be')
        print(answer)
        print(f'        Your value is')
        print(last_train[-1])
except Exception as ex:
    print(f'\n--- 0/{pts} points. Exception raised during run_k_fold_cross_validation\n')
    print(ex)

    
    



name = os.getcwd().split('/')[-1]

print()
print('='*70)
print('{} Execution Grade is {} / 90'.format(name, exec_grade))
print('='*70)


print('''\n __ / 10 Based on other testing and the results you obtain and your discussions.''')

print()
print('='*70)
print('{} FINAL GRADE is  _  / 100'.format(name))
print('='*70)

print('''
Extra Credit:

1. Add a keyword argument gpu that is False by default. If set to True, move data and 
   neural network model to the GPU and train and test it there. Compare total execution
    time without and with the use of the GPU.

2. Find another data set, not used previously in an assignment or in lecture, and repeat
   your training experiment without and with GPU on this new data.''')


print('\n{} EXTRA CREDIT is 0 / 2'.format(name))

if run_my_solution:
    # print('##############################################')
    # print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    # print('##############################################')
    pass

