# Distributed Random Forest Implementation with OpenMP and MPI

This repository contains a C++ implementation of the Random Forest algorithm in three versions:
- **Sequential**
- **Parallel using OpenMP**
- **Distributed using MPI**

The project was developed to compare performance (execution time) and test accuracy between the sequential and parallel (both shared-memory and distributed) implementations using a preprocessed version of the Adult dataset.

## Project Overview

The Random Forest implementation builds an ensemble of decision trees, for example main .cpp files are using the following hyperparameters:
- **Criterion:** Gini impurity
- **Feature selection:** "log2" (i.e., the number of features chosen equals $\log_2(\text{total features})$)
- **Maximum tree depth:** 3
- **Minimum samples to split:** 150
- **Minimum samples in leaf:** 1
- **Samples per tree:** 1,000,000

The dataset used consists of 30,000 training samples and 1,000 test samples with 107 features. The C++ code reads data files in a sparse text format where each line starts with the label followed by nonzero features in the format `index:value`.

## Experimental Results

The models were evaluated using different ensemble sizes (number of trees). Below is a summary of the results (times converted to seconds with two decimal places):

### Table 1. Execution Time and Speedup (relative to sequential)
| Number of Trees | Sequential (s) | OpenMP (s) | MPI (s) | Speedup OpenMP | Speedup MPI |
|-----------------|----------------|------------|---------|----------------|-------------|
| 100             | 27.25          | 4.55       | 5.74    | 6.00           | 4.75        |
| 500             | 119.28         | 20.24      | 31.54   | 5.89           | 3.78        |
| 1000            | 251.89         | 33.57      | 47.01   | 7.50           | 5.36        |
| 2000            | 588.34         | 68.20      | 76.22   | 8.63           | 7.72        |

### Table 2. Test Accuracy
| Number of Trees | Sequential | OpenMP | MPI  |
|-----------------|------------|--------|------|
| 100             | 0.93       | 0.94   | 0.89 |
| 500             | 0.92       | 0.95   | 0.85 |
| 1000            | 0.93       | 0.95   | 0.97 |
| 2000            | 0.96       | 0.98   | 0.97 |

### Graph

A graph (see report) shows the dependency of execution time on the number of trees for each implementation.

## How to Build and Run

### Requirements

- **Operating System:** macOS (M2 Silicon tested)
- **Compilers:**
  - For Sequential and OpenMP versions: `g++`
  - For MPI version: an MPI compiler wrapper (e.g., `mpic++`)
- **Libraries:** OpenMP (if using the OpenMP version), MPI implementation (e.g., MPICH or OpenMPI)

### Compilation Commands

- **Sequential Version:**
  ```bash
  g++ -O2 -std=c++17 sequential.cpp -o sequential
  ```

- **OpenMP Version:**
  ```bash
  g++ -O2 -fopenmp -std=c++17 openmp.cpp -o openmp 
  ```

- **MPI Version:**
  ```bash
  mpic++ -O2 -std=c++17 mpi.cpp -o mpi
  ```

### Running the Executables

- **Sequential:**
  ```bash
  ./sequential
  ```

- **OpenMP:**  
  Set the number of threads using the `OMP_NUM_THREADS` environment variable. For example:
  ```bash
  export OMP_NUM_THREADS=4
  ./openmp
  ```

- **MPI:**  
  Run with the desired number of processes. For example, to run with 4 processes:
  ```bash
  mpirun -np 4 ./mpi
  ```


## License

This project is released under the MIT License.

```
