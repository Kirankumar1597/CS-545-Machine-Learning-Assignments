import numpy as np
import optimizers as opt
import sys  

######################################################################
## class NeuralNetwork()
######################################################################

class NeuralNetwork():


    def __init__(self, n_inputs, n_hidden_units_by_layers, n_outputs):
        '''
        n_inputs: int
        n_hidden_units_by_layers: list of ints, or empty
        n_outputs: int
        '''

        self.n_inputs = n_inputs
        self.n_hidden_units_by_layers = n_hidden_units_by_layers
        self.n_outputs = n_outputs

        # Build list of shapes for weight matrices in each layera
        shapes = []
        n_in = n_inputs
        for nu in self.n_hidden_units_by_layers + [n_outputs]:
            shapes.append((n_in + 1, nu))
            n_in = nu

        self.all_weights, self.Ws = self._make_weights_and_views(shapes)
        self.all_gradients, self.Grads = self._make_weights_and_views(shapes)

        self.total_epochs = 0
        self.error_trace = []
        self.X_means = None
        self.X_stds = None
        self.T_means = None
        self.T_stds = None


    def _make_weights_and_views(self, shapes):
        '''
        shapes: list of pairs of ints for number of rows and columns
                in each layer
        Returns vector of all weights, and views into this vector
                for each layer
        '''
        all_weights = np.hstack([np.random.uniform(size=shape).flat
                                 / np.sqrt(shape[0])
                                 for shape in shapes])
        # Build list of views by reshaping corresponding elements
        # from vector of all weights into correct shape for each layer.
        views = []
        first_element = 0
        for shape in shapes:
            n_elements = shape[0] * shape[1]
            last_element = first_element + n_elements
            views.append(all_weights[first_element:last_element]
                         .reshape(shape))
            first_element = last_element

        return all_weights, views
    

    
    def __repr__(self):
        return f'NeuralNetwork({self.n_inputs}, ' + \
            f'{self.n_hidden_units_by_layers}, {self.n_outputs})'


    
    def __str__(self):
        s = self.__repr__()
        if self.total_epochs > 0:
            s += f'\n Trained for {self.total_epochs} epochs.'
            s += f'\n Final standardized training error {self.error_trace[-1]:.4g}.'
        return s
    
 

    def train(self, X, T, n_epochs, method='sgd', learning_rate=0.001, verbose=True):
        '''
        X: n_samples x n_inputs matrix of input samples, one per row
        T: n_samples x n_outputs matrix of target output values,
            one sample per row
        n_epochs: number of passes to take through all samples
            updating weights each pass
        method: 'sgd', 'adam', or 'scg'
        learning_rate: factor controlling the step size of each update
        '''

        # Setup standardization parameters
        # Setup standardization parameters
        if self.X_means is None:
            self.X_means = X.mean(axis=0)
            self.X_stds = X.std(axis=0)
            self.X_stds[self.X_stds == 0] = 1
            self.T_means = T.mean(axis=0)
            self.T_stds = T.std(axis=0)

        # Standardize X and T
        X = (X - self.X_means) / self.X_stds
        T = (T - self.T_means) / self.T_stds

        # Instantiate Optimizers object by giving it vector of all weights
        optimizer = opt.Optimizers(self.all_weights)

        _error_convert_f = lambda err: (np.sqrt(err) * self.T_stds)[0]

        if method == 'sgd':

            error_trace = optimizer.sgd(self._error_f, self._gradient_f,
                                        fargs=[X, T], n_epochs=n_epochs,
                                        learning_rate=learning_rate,
                                        error_convert_f=_error_convert_f,
                                        verbose=verbose)

        elif method == 'adam':

            error_trace = optimizer.adam(self._error_f, self._gradient_f,
                                         fargs=[X, T], n_epochs=n_epochs,
                                         learning_rate=learning_rate,
                                         error_convert_f=_error_convert_f,
                                         verbose=verbose)

        elif method == 'scg':

            error_trace = optimizer.scg(self._error_f, self._gradient_f,
                                        fargs=[X, T], n_epochs=n_epochs,
                                        error_convert_f=_error_convert_f,
                                        verbose=verbose)

        else:
            raise Exception("method must be 'sgd', 'adam', or 'scg'")

        self.total_epochs += len(error_trace)
        self.error_trace += error_trace

        # Return neural network object to allow applying other methods
        # after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)

        return self


    def _forward(self, X):
        '''
        X assumed to be standardized and with first column of 1's
        '''
        self.Ys = [X]
        for W in self.Ws[:-1]:  # forward through all but last layer
            self.Ys.append(np.tanh(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
        last_W = self.Ws[-1]
        self.Ys.append(self.Ys[-1] @ last_W[1:, :] + last_W[0:1, :])
        return self.Ys
    


    # Function to be minimized by optimizer method, mean squared error
    def _error_f(self, X, T):
        Ys = self._forward(X)
        mean_sq_error = np.mean((T - Ys[-1]) ** 2)
        return mean_sq_error
    


    # Gradient of function to be minimized for use by optimizer method
    def _gradient_f(self, X, T):
        # Assumes forward_pass just called with layer outputs saved in self.Ys.
        n_samples = X.shape[0]
        n_outputs = T.shape[1]

        # D is delta matrix to be back propagated
        D = -(T - self.Ys[-1]) / (n_samples * n_outputs)
        self._backpropagate(D)

        return self.all_gradients
    
    

    def _backpropagate(self, D):
        # Step backwards through the layers to back-propagate the error (D)
        n_layers = len(self.n_hidden_units_by_layers) + 1
        for layeri in range(n_layers - 1, -1, -1):
            # gradient of all but bias weights
            self.Grads[layeri][1:, :] = self.Ys[layeri].T @ D
            # gradient of just the bias weights
            self.Grads[layeri][0:1, :] = np.sum(D, axis=0)
            # Back-propagate this layer's delta to previous layer
            if layeri > 0:
                D = D @ self.Ws[layeri][1:, :].T * (1 - self.Ys[layeri] ** 2)
                
            

    def use(self, X):
        '''X assumed to not be standardized'''
        # Standardize X
        X = (X - self.X_means) / self.X_stds
        Ys = self._forward(X)
        # Unstandardize output Y before returning it
        return Ys[-1] * self.T_stds + self.T_means
    

    def get_error_trace(self):
        return self.error_trace
        
######################################################################
## class NeuralNetworkClassifier()
######################################################################


class NeuralNetworkClassifier(NeuralNetwork):
    

    
    def makeIndicatorVars(self,T):
        # Make sure T is two-dimensional. Should be nSamples x 1.
        if T.ndim == 1:
            T = T.reshape(-1, 1) # Reshaping the target column as 1 
        return (T == np.unique(T)) #Displaying the unique values of T
        
    #Takes target T containing labels for each class
    #Returns target T, a column for each class, a row for each sample, 1's in each row for the identified class, and 0’s for the other classes

       
        
    #Overriding the base class function
    def __str__(self):
        return NeuralNetwork.__str__(self)
        s = self.__repr__()
        if self.total_epochs > 0:
            s += f'\n Trained for {self.total_epochs} epochs.'
            s += f'\n Final standardized training error {self.error_trace[-1]:.4g}.'
        return s
       

    #Unique verison of forward used in _neg_log_likelihood_f() function call to calculate a weighted sum for the final layer
    def _softmax(self, Y):
        '''Apply to final layer weighted sum outputs'''
        # Trick to avoid overflow
        maxY = Y.max()  #Maximum value obtained in Y    
        expY = np.exp(Y - maxY) #Exponential value to power of difference between Y values and maximum value
        denom = expY.sum(1).reshape((-1, 1)) #Add one to it and reshape it to column matrix
        Y = expY / (denom + sys.float_info.epsilon) #Calcalating the delta value
        return Y #Returns Y after performing the eY etc. (softmax) calculations.
    
     

    def _neg_log_likelihood_f(self, X, T):
        #Likelihood function is the product of probability distribution function,
        Ys = self._forward(X) #Last Layer which is basically output layer
        Y = self._softmax(Ys[-1]) #Softmax function applied to only the last layer of the network. 
        #log_likelihood can be calculated with the help oof numpy.log value from Y and sys.float_info.epsilon
        log_likelihood = (np.log(Y + sys.float_info.epsilon)) #sys.float_info.epsilon equivalent to 2^-52.
        K=(T * (log_likelihood)) #N*K
        return -(np.mean(K)) #Mean value 
    
   
    def _gradient_f(self, X, T):
        n_s = X.shape[0] #X.shape[0] gives the first element in that tuple
        n_o = T.shape[1] #T.shape[1] gives the last element in that tuple
        Ys=self._forward(X) #We will get the standarized Y value from forward function to get the ;ast layer output
        Y = self._softmax(Ys[-1]) #Softmax function applied to only the last layer of the network. 
        D = -(T - Y)/(n_s * n_o) #Calculating delta value
        self._backpropagate(D) #Call backpropagating function and pass in delta value
        return self.all_gradients #Return our all_gradients
    
     
       
    def train(self, X, T, n_epochs, method='sgd', learning_rate=None, verbose=True):
        '''
        X: n_samples x n_inputs matrix of input samples, one per row
        T: n_samples x n_outputs matrix of target output values,
            one sample per row
        n_epochs: number of passes to take through all samples
            updating weights each pass
        method: 'sgd', 'adam', or 'scg'
        learning_rate: factor controlling the step size of each update
        '''

        # Standardization parameters
        if self.X_means is None:
            self.X_means = X.mean(axis=0) #Mean of X values
            self.X_stds = X.std(axis=0) #Standand Deviation of X values
            self.X_stds[self.X_stds == 0] = 1 #Standand Deviation of X values to be in 1 if the value is zero
            self.T_means = T.mean(axis=0) #Mean of T values
            self.T_stds = T.std(axis=0) #Standand Deviation of T values

        # Standardize X using standarization formula
        X = (X - self.X_means) / self.X_stds
        
        #self.class is something that should be assigned inside the train function because that is the first time of our function have seen the target value. And we can use np.unique to get the unique values from that. 
        self.classes =np.unique(T)
        # Convert targets into indicator variables with the help of akeIndicatorVars function. 
        T_ind = self.makeIndicatorVars(T)  
       
        # The idea is to calculate the gradients and update the weights. One way to update the weights here once we have the optimizers defined above is to take a step down that gradient.  
        optimizer = opt.Optimizers(self.all_weights)

        _error_convert_f = lambda err: np.exp(-err)

        #Case 1: method: sgd
        if method == 'sgd':

            error_trace = optimizer.sgd(self._neg_log_likelihood_f, self._gradient_f,
                                        fargs=[X, T_ind], n_epochs=n_epochs,
                                        learning_rate=learning_rate,
                                        error_convert_f=_error_convert_f,
                                        verbose=verbose)
        #Case 2: method: adam
        elif method == 'adam':

            error_trace = optimizer.adam(self._neg_log_likelihood_f, self._gradient_f,
                                         fargs=[X, T_ind], n_epochs=n_epochs,
                                         learning_rate=learning_rate,
                                         error_convert_f=_error_convert_f,
                                         verbose=verbose)
        #Case 3: method: scg
        elif method == 'scg':

            error_trace = optimizer.scg(self._neg_log_likelihood_f, self._gradient_f,
                                        fargs=[X, T_ind], n_epochs=n_epochs,
                                        error_convert_f=_error_convert_f,
                                        verbose=verbose)

        #Default
        else:
            raise Exception("method must be 'sgd', 'adam', or 'scg'")

        #Total number of epochs to be length of the error_trace calculated above
        self.total_epochs += len(error_trace)
        #Storing it in the object
        self.error_trace += error_trace

        #Return the object 
        return self 
    
    
       
    def use(self, X): #We have our inpupt X coming in
        X_std = (X - self.X_means) / self.X_stds  #standardize X value
        Y = self._forward(X_std)  #Calling the forward function as we do calculation through all the layer and we will get Y which would be  standardized. So we don't need to use the standardization formula as we used it on X. 
        probs = self._softmax(Y[-1]) #If we are doing classifiacation in our neural network, we have to pass all those values to softmax function.So these are the actual probablities between zero and one.
        # We have to find the maximum value in each row, because we have as many rows as we have samples and columns as we have classes which is same as number of outputs of our neural network. So, for each row, we want to find which one is the largest.   
        # We don't need to use for loop for maximum value as np.agrmax does it for whole matrix probs. Giving us the column index that is the maximum value in each row separately.  
        classes = (self.classes[np.argmax(probs,axis=1)]).reshape(-1,1) 
        #The reason for reshape is to make our class to be a column matrix. 
        return classes, probs #Return classes, probs
