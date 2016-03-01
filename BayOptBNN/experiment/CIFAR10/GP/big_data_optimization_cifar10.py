# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
This is a simple demo to demonstrate the use of Bayesian optimization with GPyOpt with using sparse GPs. Run the example by writing:

import GPyOpt
BO_demo_big_data = GPyOpt.demos.big_data_optimization()

As a result you should see:

- A plot with the model and the current acquisition function
- A plot with the diagnostic plots of the optimization.
- An object call BO_demo_auto that contains the results of the optimization process (see reference manual for details). Among the available results you have access to the GP model via

>> BO_demo_big_data.model

and to the location of the best found location writing.

BO_demo_big_data.x_opt

"""
import os, sys
project_dir = os.environ['EXPERI_PROJECT_PATH']
sys.path.append(project_dir)
sys.path.append(project_dir+"/library/GPyOpt")
sys.path.append(project_dir+"/hyperParamServerSubSet")
sys.path.append(project_dir+"/library")
sys.path.append(project_dir+"/library/autogradwithbay")
sys.path.append(project_dir+"/library/hypergrad")
sys.path.append(project_dir+"/library/autograd")
sys.path.append(project_dir+"/library/spearmint")
sys.path.append(project_dir+"/library/spearmint/spearmint")

def big_data_optimization(plots=True):
    import GPyOpt
    from numpy.random import seed
    seed(12345)
    
    # --- Objective function
    objective_noisy = GPyOpt.fmodels.experimentCIFAR10.cifar10Experiment(input_dim = 4,bounds = [(0.001,0.1),(0.6,0.9),(0.2,0.7),(0.2,0.7)])
    bounds = objective_noisy.bounds

    # --- Problem definition and optimization
    BO_demo_big_data = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,  # function to optimize       
                                            bounds=bounds,                 # box-constrains of the problem
                                            acquisition='LCB',             # Selects the Lower Confidence Bound criterion
                                            acquisition_par = 2,           # parameter of the acquisition function
                                            normalize = True,              # normalized acquisition function      
                                            sparseGP = True,               # Use a sparse GP for the sparse GP.
                                            num_inducing = 10,             # Number of inducing points
                                            numdata_initial_design = 10)        # Initialize the model with 1000 points
    
    # Run the optimization
    max_iter = 150

    print '-----'
    print '----- Running demo. It may take a few seconds.'
    print '-----'
    
    # --- Run the optimization                                              # evaluation budget
    BO_demo_big_data.run_optimization(max_iter,                             # Number of iterations
                                acqu_optimize_method = 'fast_random',       # method to optimize the acq. function
                                acqu_optimize_restarts = 30,                # number of local optimizers
                                eps = 10e-6,                                # secondary stop criteria (apart from the number of iterations) 
                                true_gradients = False)                     # The gradients of the acquisition function are approximated (faster)
    # --- Plots
    if plots:
        BO_demo_big_data.plot_convergence("gp_covergence_final.pdf")
    BO_demo_big_data.save_report("gp_report_final_inputdim1000.pdf")
    BO_demo_big_data.save_result()
    
    return BO_demo_big_data
if __name__=='__main__':
    big_data_optimization()
