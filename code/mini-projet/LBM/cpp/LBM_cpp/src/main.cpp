#include <cstdlib>
#include <string>
#include <iostream>

#include "lbm/LBMSolver.h" 
  // adding timer
#include <utils/monitoring/SimpleTimer.h>

// TODO : uncomment when building with OpenACC
//#include "utils/openacc_utils.h"

int main(int argc, char* argv[])
{

  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = argc>1 ? std::string(argv[1]) : "flowAroundCylinder.ini";

  // TODO : uncomment the last two lines when activating OpenACC
  // print OpenACC version / info
  // print_openacc_version();
  //init_openacc();

  ConfigMap configMap(input_file);

  // test: create a LBMParams object
  LBMParams params = LBMParams();
  params.setup(configMap);

  // print parameters on screen
  params.print();

  LBMSolver* solver = new LBMSolver(params);

// using timer
  SimpleTimer timer = SimpleTimer();
  timer.start();
  // std::cout << "Ici" << std::endl;
  solver->run();
  // std::cout << "Là" << std::endl;
  timer.stop();
  std::cout << timer.elapsed() << std::endl;

  const int nx = params.nx;
  const int ny = params.ny;
  const int maxIter = params.maxIter;

  printf("Results;%d;%d;%f\n", nx*ny, maxIter, timer.elapsed());

  delete solver;

  return EXIT_SUCCESS;
}
