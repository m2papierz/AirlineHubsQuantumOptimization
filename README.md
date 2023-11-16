# AirlineHubsQuantumOptimization

A fun project aiming to optimize the airline's operating costs by strategically selecting specific airports as hubs to 
minimize costs using quantum annealer. The problem in the project is defined in two ways - as a Discrete Quadratic Model 
and as a Constrained Quadratic Model.

The original problem was presented in:  
[O'Kelly, Morton. (1987). A Quadratic Integer Program for the Location of Interacting Hub Facilities.
](https://www.researchgate.net/publication/221990142_A_Quadratic_Integer_Program_for_the_Location_of_Interacting_Hub_Facilities)

The derivation of the formulas needed to formulate the quadratic model was based on the tutorial:  
[https://github.com/dwave-examples/airline-hubs](https://github.com/dwave-examples/airline-hubs)

## Project setup

A D-Wave Leap account is required to access quantum annealer: https://cloud.dwavesys.com/leap/

To install all necessary packages, run this line in command line:  
`pip install -r requirements.txt`

## Project execution

1. Set the parameters in the configuration file `config.yaml`.
2. Run `run_solver.py`.
3. The resulting optimal solution is located in the `reports` folder.