#include <cstdlib>
#include <string>
#include <iostream>

// =========================
// CUDA imports 
// =========================
#include <cuda_runtime.h>
  // adding timer
#include <utils/monitoring/CudaTimer.h>


#include "lbm/LBMSolver.h" 

int main(int argc, char* argv[])
{

  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = argc>1 ? std::string(argv[1]) : "flowAroundCylinder.ini";

  ConfigMap configMap(input_file);

  // create a LBMParams object
  LBMParams params = LBMParams();
  params.setup(configMap);

  // print parameters on screen
  params.print();

  LBMSolver* solver = new LBMSolver(params);

  // using timer
  CudaTimer gpuTimer = CudaTimer();
  gpuTimer.start();
  // std::cout << "Ici" << std::endl;
  solver->run();
  // std::cout << "Là" << std::endl;
  gpuTimer.stop();
  std::cout << gpuTimer.elapsed() << std::endl;

  const int nx = params.nx;
  const int ny = params.ny;
  const int npop = LBMParams::npop;
  const int maxIter = params.maxIter;

  // 1555.2 because: Memory Clock Rate 1215Mhz (/ 1000 to get Ghz) * (Memory Bus Width (5120bit / 8 channels) * 2 because reading and writing)
  // cudaDeviceProp deviceProp; To obtain the values of material properties
  // float max_bandwidth = 1215 / 1000 * 5120 / 8 * 2;
  
  // Number of lattice nodes * ((4 reading * 9 because of npop) + 3 writing) * (1e-9 * 8 because of float bits size / time)
  // float macroscopic_kernel_cost = nx * ny * ((4 * 9) + 3) * 8;// * 1e-9 / gpuTimer.elapsed();
  
  // float equilibrium_kernel_cost = nx * ny * ((4 + 1) * 9) * 8;// * 1e-9 / gpuTimer.elapsed();

  // float border_outflow_kernel_cost = ny * ((1 + 1)) * 8;// * 1e-9 / gpuTimer.elapsed();

  // float border_inflow_kernel_cost = ny * ((1 + 1)) * 8;// * 1e-9 / gpuTimer.elapsed();

  // float update_fin_inflow_kernel_cost = ny * ((1 + 1)) * 8;// * 1e-9 / gpuTimer.elapsed();

  // float compute_collision_kernel_cost = nx * ny * ((1 + 1)) * 8;// * 1e-9 / gpuTimer.elapsed();

  // float update_obstacle_kernel_cost = nx * ny * (1 + (2 + 1) * 9) * 8;// * 1e-9 / gpuTimer.elapsed();

  // float streaming_kernel_cost = nx * ny * ((2 + 1) * 9) * 8;// * 1e-9 / gpuTimer.elapsed();
  
  
  // float total_cost = maxIter * 1e-9 / gpuTimer.elapsed() * (macroscopic_kernel_cost + equilibrium_kernel_cost + border_outflow_kernel_cost + border_inflow_kernel_cost + update_fin_inflow_kernel_cost + compute_collision_kernel_cost + update_obstacle_kernel_cost + streaming_kernel_cost);
  
  
  // As we read and write on all our device variable on each iteration, the total cost of an iteration is:
  // Total cost is: (Three fin,fout,feq arrays of real_t on nx * ny * npop) + (Three rho,ux,uy arrays of real_t on nx * ny) + (One array obstacle of int on nx * ny)
  float total_cost = sizeof(real_t) * ( 3 * nx * ny * npop + 3 * nx * ny) + sizeof(int) * nx * ny;
  // Number of iterations * to get a result in giga * read and write / time elapsed * size of computing
  float GFLOPs = maxIter * 1e-9 * 2 / gpuTimer.elapsed() * (total_cost);
  
  std::cout << "Total cost (in GFLOP/s) : " << GFLOPs << std::endl;
  std::cout << "time : " << gpuTimer.elapsed() << std::endl;
  std::cout << "sizeof(real_t) : " << sizeof(real_t) << std::endl;

  // printf("Parser;%d;%d;%f;%f;%3.5f;%f\n", nx*ny, maxIter,gpuTimer.elapsed(),4,5,total_cost);
  printf("Results;%d;%d;%f;%f\n", nx*ny, maxIter, gpuTimer.elapsed(), GFLOPs);

  delete solver;

  return EXIT_SUCCESS;
}
