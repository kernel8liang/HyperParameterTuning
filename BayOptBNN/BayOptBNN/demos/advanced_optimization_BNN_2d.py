# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
This is a simple demo to demonstrate the use of Bayesian optimization with BO_BNN with some simple options. Run the example by writing:

import BO_BNN
BO_demo_2d = BO_BNN.demos.advanced_optimization_2d()

As a result you should see:

- A plot with the model and the current acquisition function
- A plot with the diagnostic plots of the optimization.
- An object call BO_demo_auto that contains the results of the optimization process (see reference manual for details). Among the available results you have access to the GP model via

>> BO_demo_2d.model

and to the location of the best found location writing.

BO_demo_2d.x_opt

"""

def advanced_optimization_2d(plots=True):
    import BayOptBNN
    from numpy.random import seed
    seed(12345)
    

    # --- Objective function
    objective_true  = BayOptBNN.fmodels.experiments2d.sixhumpcamel()             # true function
    # objective_true.plot()
    objective_true.plot()
    objective_noisy = BayOptBNN.fmodels.experiments2d.sixhumpcamel(sd = 0.1)     # noisy version
    bounds = objective_noisy.bounds                                           # problem constrains 
    input_dim = len(bounds)



    # --- Problem definition and optimization
    BO_demo_2d = BayOptBNN.methods.BayesianOptimizationBNN(f=objective_noisy.f,  # function to optimize
                                            bounds=bounds,                 # box-constrains of the problem
                                            acquisition='LCB',             # Selects the Expected improvement
                                            acquisition_par = 2,           # parameter of the acquisition function
                                            numdata_initial_design = 15,    # 15 initial points   
                                            type_initial_design='latin',   # latin desing of the initial points 
                                            model_optimize_interval= 2,    # The model is updated every two points are collected
                                            layer_sizes =[2, 10, 10, 1],
                                            normalize = True)              # normalized y                       
    
    
    # Run the optimization
    max_iter = 70

    print '-----'
    print '----- Running demo. It may take a few seconds.'
    print '-----'
    
    # --- Run the optimization                                              # evaluation budget
    BO_demo_2d.run_optimization(max_iter,                                   # Number of iterations
                                acqu_optimize_method = 'DIRECT',       # method to optimize the acq. function
                                acqu_optimize_restarts = 5,                # number of local optimizers
                                eps=10e-6,                        # secondary stop criteria (apart from the number of iterations) 
                                true_gradients = True)                     # The gradients of the acquisition function are approximated (faster)
   

    # --- Plots
    if plots:
        objective_true.plot()
        BO_demo_2d.plot_acquisition("bnn_acquisition_final.pdf")
        BO_demo_2d.plot_convergence("bnn_covergence_final.pdf")
        
    
    return BO_demo_2d 

if __name__=='__main__':
    advanced_optimization_2d()