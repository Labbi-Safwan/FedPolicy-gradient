# Fed-PG
To run the experiments, you need to specify the algorithm and the environment from the command line:
 --environment (0 for synthetic and 1 for GridWord), --alg (0 is for SoftFedPG, 1 is for RegSoftFedPG, 2 for BitRegSoftFedPG, and 3 for FedQlearning )
 The hetereogeneity on the transition kernel, the number of runs, the number of rounds, of local steps, the sterp size, and the remaining parameters can be manually set inside the file. 

To run the simulation
```
	# for the simulation itself
	python main.py 
```
To make the plots you need to modify the corresponding paramters in the plot_experiments.py file
and specify from the command line: --environment (0 for synthetic and 1 for GridWord) and --experiment (from 1 to 3).

```
	# to make the plots of the paper
	python plots_main.py 
```


