""" This class use the bayesian optimization by Gaussian Process to tune the median of the hyperparameters for the weights,
    Then Gradient based method is used to tune hyperparameters for the weights locally """

import optparse
import tempfile
import datetime
import multiprocessing
import importlib
import time
import imp
import os
import sys
import re
import signal
import socket

try: import simplejson as json
except ImportError: import json

from os import listdir
from os.path import isfile, join


# TODO: this shouldn't be necessary when the project is installed like a normal
# python lib.  For now though, this lets you symlink to supermint from your path and run it
# from anywhere.
sys.path.append(os.path.realpath(__file__))
project_dir = os.environ['EXPERI_PROJECT_PATH']
sys.path.append(project_dir)
sys.path.append(project_dir+"/hyperParamServerSubSet")
sys.path.append(project_dir+"/library/spearmint")
sys.path.append(project_dir+"/library/spearmint/spearmint")

from spearmint.ExperimentGrid  import *
from spearmint.helpers         import *
from spearmint.runner          import job_runner
import sys
import random
import numpy as np

# Use a global for the web process so we can kill it cleanly on exit
web_proc = None

Flag = True

def parse_args():
    parser = optparse.OptionParser(usage="\n\tspearmint [options] <experiment/config.pb>")

    parser.add_option("--max-concurrent", dest="max_concurrent",
                      help="Maximum number of concurrent jobs.",
                      type="int", default=1)
    parser.add_option("--max-finished-jobs", dest="max_finished_jobs",
                      type="int", default=100)
    parser.add_option("--method", dest="chooser_module",
                      help="Method for choosing experiments [SequentialChooser, RandomChooser, GPEIOptChooser, GPEIOptChooser, GPEIperSecChooser, GPEIChooser]",
                      type="string", default="GPEIOptChooser")
    parser.add_option("--driver", dest="driver",
                      help="Runtime driver for jobs (local, or sge)",
                      type="string", default="local")
    parser.add_option("--method-args", dest="chooser_args",
                      help="Arguments to pass to chooser module.",
                      type="string", default="")
    parser.add_option("--grid-size", dest="grid_size",
                      help="Number of experiments in initial grid.",
                      type="int", default=20000)
    parser.add_option("--grid-seed", dest="grid_seed",
                      help="The seed used to initialize initial grid.",
                      type="int", default=1)
    parser.add_option("--run-job", dest="job",
                      help="Run a job in wrapper mode.",
                      type="string", default="")
    parser.add_option("--polling-time", dest="polling_time",
                      help="The time in-between successive polls for results.",
                      type="float", default=3.0)
    parser.add_option("-w", "--web-status", action="store_true",
                      help="Serve an experiment status web page.",
                      dest="web_status")
    parser.add_option("--port",
                      help="Specify a port to use for the status web interface.",
                      dest="web_status_port", type="int", default=0)
    parser.add_option("--host",
                      help="Specify a host to use for the status web interface.",
                      dest="web_status_host", type="string", default=None)
    parser.add_option("-v", "--verbose", action="store_true",
                      help="Print verbose debug output.")
    parser.add_option("--mode", dest="mode_operate",
                      help="the mode to choose (generate or get from output)")


    (options, args) = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
        sys.exit(0)

    return options, args


def get_available_port(portnum):
    if portnum:
        return portnum
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', portnum))
    port = sock.getsockname()[1]
    sock.close()
    return port


def start_web_view(options, experiment_config, chooser):
    '''Start the web view in a separate process.'''

    from spearmint.web.app import app
    port = get_available_port(options.web_status_port)
    print "Using port: " + str(port)
    if options.web_status_host:
        print "Listening at: " + str(options.web_status_host)
    app.set_experiment_config(experiment_config)
    app.set_chooser(options.chooser_module,chooser)
    debug = (options.verbose == True)
    start_web_app = lambda: app.run(debug=debug, port=port, host=options.web_status_host)
    proc = multiprocessing.Process(target=start_web_app)
    proc.start()

    return proc


def main():
    (options, args) = parse_args()

    project_dir = os.environ['EXPERI_PROJECT_PATH']
    sys.path.append(project_dir)
    sys.path.append(project_dir+"/library/spearmint")

    log("experiment_config is: " + args[0]);

    if options.job:
        job_runner(load_job(options.job))
        log("run times are: ");
        exit(0)



    experiment_config = args[0]
    expt_dir  = os.path.dirname(os.path.realpath(experiment_config))
    log("Using experiment configuration: " + experiment_config)
    log("experiment dir: " + expt_dir)

    if not os.path.exists(expt_dir):

        log("Cannot find experiment directory '%s'. "
            "Aborting." % (expt_dir))
        sys.exit(-1)

    check_experiment_dirs(expt_dir)



    # Generate the yaml file
    classIndexFile = os.path.join(expt_dir, 'classIndex.txt')
    if not os.path.exists(classIndexFile):
        with open(classIndexFile, 'w') as fout:
            classIndex=random.sample(xrange(0,99), 20)
            fout.write("\t".join(np.array(map(str, classIndex))))

    # Load up the chooser module.
    module  = importlib.import_module('spearmint.chooser.' + options.chooser_module)
    chooser = module.init(expt_dir, options.chooser_args)

    log("module is: "+options.chooser_module);

    if options.web_status:
        web_proc = start_web_view(options, experiment_config, chooser)

    # Load up the job execution driver.
    module = importlib.import_module('spearmint.driver.' + options.driver)
    driver = module.init()

    #mode for user to choose: generate and test

    mode = options.mode_operate
    #"generate"
    #mode = "yaml_to_hyper"
  #  mode = "get_hyper_from_output"




    if mode == "yaml_to_hyper":
        # experiment_dir = "/mnt/random/hyperopt_experiment_1"
        yamlsDirectoryPath = expt_dir + "/hyper_yamels"

        hyper_file = os.path.join(expt_dir, 'hyperyaml.yaml')
        hyperList = []

        number = listdir( yamlsDirectoryPath).__len__()
        print("there are total "+str(number)+" yamls in the hyper_yaml directory")
        count = 0

        with open(hyper_file, 'r') as fin:
            for line in fin:
                if '!hyperopt' in line:
                    hyperList.append( line.split()[0] + " " + line.split()[2] + " " + line.split()[3])

        checkstatus = True;
        while  checkstatus:
            checkstatus = attempt_yaml_to_hyper(experiment_config, expt_dir, driver, options, count, hyperList, yamlsDirectoryPath,number,mode)
            if Flag:
                count = count+1
                global Flag
                Flag = True

            time.sleep(options.polling_time*2)
            #time.sleep(2)

    elif mode == "get_hyper_from_output":
        # experiment_dir = "/mnt/random/hyperopt_experiment_1"
        yamlsDirectoryPath = expt_dir + "/output_pre"

        hyper_file = os.path.join(expt_dir, 'hyperyaml.yaml')
        hyperList = []

#        number = listdir( yamlsDirectoryPath).__len__()
        number = options.max_finished_jobs
        print("there are total "+str(number)+" yamls in the hyper_yaml directory")
        count = 0

        with open(hyper_file, 'r') as fin:
            for line in fin:
                if '!hyperopt' in line:
                    hyperList.append( line.split()[0] + " " + line.split()[2] + " " + line.split()[3])

        checkstatus = True;
        while  checkstatus:
            checkstatus = attempt_yaml_to_hyper(experiment_config, expt_dir, driver, options, count, hyperList, yamlsDirectoryPath,number,mode)
            if Flag:
                count = count+1
                global Flag
                Flag = True

            time.sleep(options.polling_time*2)
    # Loop until we run out of jobs.
    elif mode == "generate":

        while attempt_dispatch(experiment_config, expt_dir, chooser, driver, options):
            # This is polling frequency. A higher frequency means that the algorithm
            # picks up results more quickly after they finish, but also significantly
            # increases overhead.
            time.sleep(options.polling_time)

    else:
        exit(0)

# TODO:
#  * move check_pending_jobs out of ExperimentGrid, and implement two simple
#  driver classes to handle local execution and SGE execution.
#  * take cmdline engine arg into account, and submit job accordingly
def attempt_yaml_to_hyper(expt_config, expt_dir,  driver, options, count, hyperList, yamlsDirectoryPath, number,mode):



    log("\n" + "-" * 40)
    expt = load_experiment(expt_config)

    # Build the experiment grid.
    expt_grid = ExperimentGrid(expt_dir,
                               expt.variable,
                               options.grid_size,
                               options.grid_seed)

    # Print out the current best function value.
    best_val, best_job = expt_grid.get_best()
    if best_job >= 0:
        log("Current best: %f (job %d)" % (best_val, best_job))
    else:
        log("Current best: No results returned yet.")


    # Returns lists of indices.
    pending    = expt_grid.get_pending()
    complete   = expt_grid.get_complete()


    n_pending    = pending.shape[0]
    n_complete   = complete.shape[0]

    log("  %d pending   %d complete" %
        ( n_pending, n_complete))

    if n_complete < count:
        print("  %d count   %d complete" %( count, n_complete))
        global Flag
        Flag = False
        return True

    else:
        global Flag
        Flag = True
    # Verify that pending jobs are actually running, and add them back to the
    # candidate set if they have crashed or gotten lost.
    for job_id in pending:
        proc_id = expt_grid.get_proc_id(job_id)
        if not driver.is_proc_alive(job_id, proc_id):
            log("Set job %d back to pending status." % (job_id))
            expt_grid.set_candidate(job_id)


    # Print out the best job results
    write_best_job(expt_dir, best_val, best_job, expt_grid)



    if n_complete >= number:
        log("All yaml files are read and run in full data set."
                         "Exiting" % options.max_finished_jobs)
        return False

    else:

        # choose the job-id to be the next job that read from hyper_yaml files
        job_id = count

        log("selected job %d from the grid." % (job_id))

        # Convert this back into an interpretable job and add metadata.
        job = Job()
        job.id        = job_id
        job.expt_dir  = expt_dir
        job.name      = expt.name
        job.language  = expt.language
        job.status    = 'submitted'
        job.submit_t  = int(time.time())
        # todo: read the line after the learn-rate in yamls file and reconding to the hyperopt.yaml file
        params = []

        travelFile = listdir( yamlsDirectoryPath)[ count]
        filePath = join(yamlsDirectoryPath, travelFile)

        if mode== "get_hyper_from_output":
            if isfile( filePath):

                print("the path is " + travelFile)


                with open(filePath, 'r') as fin:
                    for line in fin:
                        if 'spear_wrapper params are:' in line:
                            for i in xrange(len(hyperList)):
                                hyper_param = hyperList[i]
                                hyper_split = hyper_param.split()
                                if hyper_split[1] in line:
                                    dic = ((line.split(hyper_split[1])[1]).split(": array([")[1]).split("])")[0];
                                    param = Parameter()

                                    param.name = hyper_split[1]

                                    type = hyper_split[2]
                                    if type == "FLOAT":
                                        param.dbl_val.append (float(dic))
                                    elif type == "INT":
                                        param.int_val.append (int(dic))
                                    elif type == "ENUM":
                                        param.str_val.append (str(dic))
                                    else:
                                        param.str_val.append (str(dic))

                                    params.append(param)
                            break


                param = Parameter()
                param.name = "experiment_dir"
                param.str_val.append (expt_dir)
                params.append(param)

                param = Parameter()
                param.name = "pre_output_file_name"
                param.str_val.append (travelFile)
                params.append(param)

                job.param.extend(params)
            else :
                job.param.extend(params)
        else:
            if isfile( filePath):

                print("the path is " + travelFile)

                i = 0
                with open(filePath, 'r') as fin:
                    for line in fin:
                        if(i< hyperList.__len__()):
                            hyper_param = hyperList[i]
                            hyper_split = hyper_param.split()
                            if hyper_split[0] in line:
                                dic = [k.strip(",") for k in line.split()][1]
                                param = Parameter()

                                param.name = hyper_split[1]

                                type = hyper_split[2]
                                if type == "FLOAT":
                                    param.dbl_val.append (float(dic))
                                elif type == "INT":
                                    param.int_val.append (int(dic))
                                elif type == "ENUM":
                                    param.str_val.append (str(dic))
                                else:
                                    param.str_val.append (str(dic))

                                params.append(param)

                            i = i+1


                param = Parameter()
                param.name = "experiment_dir"
                param.str_val.append (expt_dir)
                params.append(param)

                job.param.extend(params)
            else :
                job.param.extend(params)

        save_job(job)
        pid = driver.submit_job(job)
        if pid != None:
            log("submitted - pid = %d" % (pid))
            expt_grid.set_submitted(job_id, pid)
        else:
            log("Failed to submit job!")
            log("Deleting job file.")
            os.unlink(job_file_for(job))

    return True

def attempt_dispatch(expt_config, expt_dir, chooser, driver, options):
    log("\n" + "-" * 40)
    expt = load_experiment(expt_config)

    # Build the experiment grid.
    expt_grid = ExperimentGrid(expt_dir,
                               expt.variable,
                               options.grid_size,
                               options.grid_seed)

    # Print out the current best function value.
    best_val, best_job = expt_grid.get_best()
    if best_job >= 0:
        log("Current best: %f (job %d)" % (best_val, best_job))
    else:
        log("Current best: No results returned yet.")

    # Gets you everything - NaN for unknown values & durations.
    grid, values, durations = expt_grid.get_grid()

    # Returns lists of indices.
    candidates = expt_grid.get_candidates()
    pending    = expt_grid.get_pending()
    complete   = expt_grid.get_complete()

    n_candidates = candidates.shape[0]
    n_pending    = pending.shape[0]
    n_complete   = complete.shape[0]
    log("%d candidates   %d pending   %d complete" %
        (n_candidates, n_pending, n_complete))

    # Verify that pending jobs are actually running, and add them back to the
    # candidate set if they have crashed or gotten lost.
    for job_id in pending:
        proc_id = expt_grid.get_proc_id(job_id)
        if not driver.is_proc_alive(job_id, proc_id):
            log("Set job %d back to pending status." % (job_id))
            expt_grid.set_candidate(job_id)

    # Track the time series of optimization.
    write_trace(expt_dir, best_val, best_job, n_candidates, n_pending, n_complete)

    # Print out the best job results
    write_best_job(expt_dir, best_val, best_job, expt_grid)

    if n_complete >= options.max_finished_jobs:
        log("Maximum number of finished jobs (%d) reached."
                         "Exiting" % options.max_finished_jobs)
        return False

    if n_candidates == 0:
        log("There are no candidates left.  Exiting.")
        return False

    if n_pending >= options.max_concurrent:
        log("Maximum number of jobs (%d) pending." % (options.max_concurrent))
        return True

    else:

        # start a bunch of candidate jobs if possible
        #to_start = min(options.max_concurrent - n_pending, n_candidates)
        #log("Trying to start %d jobs" % (to_start))
        #for i in xrange(to_start):

        # Ask the chooser to pick the next candidate
        log("Choosing next candidate... ")
        job_id = chooser.next(grid, values, durations, candidates, pending, complete)

        # If the job_id is a tuple, then the chooser picked a new job.
        # We have to add this to our grid
        if isinstance(job_id, tuple):
            (job_id, candidate) = job_id
            job_id = expt_grid.add_to_grid(candidate)

        log("selected job %d from the grid." % (job_id))

        # Convert this back into an interpretable job and add metadata.
        job = Job()
        job.id        = job_id
        job.expt_dir  = expt_dir
        job.name      = expt.name
        job.language  = expt.language
        job.status    = 'submitted'
        job.submit_t  = int(time.time())
        temp = expt_grid.get_params(job_id)

        param = Parameter()
        param.name = "experiment_dir"
        param.str_val.append (expt_dir)
        params=expt_grid.get_params(job_id)
        params.append(param)
        job.param.extend(params)

        #log("the params are " + expt_grid.get_params(job_id)[0]);
        save_job(job)
        pid = driver.submit_job(job)
        if pid != None:
            log("submitted - pid = %d" % (pid))
            expt_grid.set_submitted(job_id, pid)
        else:
            log("Failed to submit job!")
            log("Deleting job file.")
            os.unlink(job_file_for(job))

    return True


def write_trace(expt_dir, best_val, best_job,
                n_candidates, n_pending, n_complete):
    '''Append current experiment state to trace file.'''
    trace_fh = open(os.path.join(expt_dir, 'trace.csv'), 'a')
    trace_fh.write("%d,%f,%d,%d,%d,%d\n"
                   % (time.time(), best_val, best_job,
                      n_candidates, n_pending, n_complete))
    trace_fh.close()


def write_best_job(expt_dir, best_val, best_job, expt_grid):
    '''Write out the best_job_and_result.txt file containing the top results.'''

    best_job_fh = open(os.path.join(expt_dir, 'best_job_and_result.txt'), 'w')
    best_job_fh.write("Best result: %f\nJob-id: %d\nParameters: \n" %
                      (best_val, best_job))
    for best_params in expt_grid.get_params(best_job):
        best_job_fh.write(str(best_params))
    best_job_fh.close()


def check_experiment_dirs(expt_dir):
    '''Make output and jobs sub directories.'''

    output_subdir = os.path.join(expt_dir, 'output')
    check_dir(output_subdir)

    job_subdir = os.path.join(expt_dir, 'jobs')
    check_dir(job_subdir)

# Cleanup locks and processes on ctl-c
def sigint_handler(signal, frame):
    if web_proc:
        print "closing web server...",
        web_proc.terminate()
        print "done"
    sys.exit(0)


if __name__=='__main__':
    print "setting up signal handler..."
    signal.signal(signal.SIGINT, sigint_handler)
    main()
