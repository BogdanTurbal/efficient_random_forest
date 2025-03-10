g++ -O2 -fopenmp -std=c++17 openmp.cpp -o openmp
g++ -O2 -std=c++17 sequential.cpp -o sequential
mpic++ -O2 -std=c++17 mpi.cpp -o mpi

./sequential
./openmp
mpirun -np 12 ./mpi