# gaussian_processes_teishutu
Python implementation of a regression model using Gaussian Process. It can be executed in a virtual environment (Conda).
The learning process of the Gaussian Process Regression (GPR) will be shown as an animation using matplotlib.

## Environment Setup

Follow these steps to set up the development environment for this project:

### 1 Install Conda (if not installed)
Ensure you have **Miniconda** or **Anaconda** installed. If not, download and install Miniconda

### 2️ Create a Virtual Environment
You can create the environment using the provided `environment.yml` file:
dependencies
  - python=3.11.5
  - numpy
  - matplotlib
  - ffmpeg
    
```bash
conda env create -f environment.yml
```
inst

#　After installation, activate the environment:
　
```bash
conda activate my_gaussian_env
```

If FFmpeg does not work as expected, install it manually:
　
```bash
choco install ffmpeg
```

## Code Structure Overview
###1. Kernel class
 - BaseKernels: A base class for defining kernels with initialization and computation methods.
 - RBF: Implements the Radial Basis Function (RBF) kernel, which measures similarity between two points based on their distance.
   $$ K(X, Y) = \frac{1}{\sqrt{2 \pi} \sigma^d} \exp \left( -\frac{||X - Y||^2}{2 \sigma^2} \right) $$
 - KernelFactory: A factory to generate different kernel types (this code has only RBF).
   
###2 GP class
 - GP: A Gaussian Process regressor that models a function using a kernel. It updates with new data and makes predictions with associated uncertainty (mean and variance).

###3 Data Generator Class
 - BaseDataGenerator: A base class for generating data.
 - BasicGenerator: Generates random data points using a sine function (y = 2 * sin(x)) with added noise.
 
###4 Plotting and Animation
 - plotdata: Computes the GP's predicted mean and variance for given inputs.
 - create_movie: Creates and saves an animation showing the GP learning process over time.

###5 Main Function
 - main: Coordinates the process of generating data, training the GP model, and visualizing the results as an animated movie.
