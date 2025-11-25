# PaCMAP
This project is done to experiment with the PaCMAP DR technique on different datasets 


## Example runs 
python -u run.py coil20 pacmap all plot
python -u run.py coil20-npy pacmap none plot 3
python -u run.py mnist pacmap knn no 2
python -u run.py olivetti


## Experiment Results
### Run DR on training set only 
![MNIST](results/mnist.png)
![MNIST](results/coil20.png)
![MNIST](results/olivetti.png)