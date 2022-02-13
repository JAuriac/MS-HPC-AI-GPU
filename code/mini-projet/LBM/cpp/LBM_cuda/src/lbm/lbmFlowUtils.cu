#include <math.h> // for M_PI = 3.1415....

#include "lbmFlowUtils.h"

#include "lbmFlowUtils_kernels.h"
#include "cuda_error.h"

// ======================================================
// ======================================================
void macroscopic(const LBMParams& params, 
                 const velocity_array_t v,
                 const real_t* fin_d,
                 real_t* rho_d,
                 real_t* ux_d,
                 real_t* uy_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  // TODO : call kernel
  // int nbThreads = 8;
  // dim3 blockSize(nbThreads,1,1);
  // dim3 gridSize((N+1)/nbThreads,1,1);

  // unsigned int threadsPerBlockX=32;
  // unsigned int threadsPerBlockY=32;
  // dim3 blockSize(threadsPerBlockX,threadsPerBlockY,1);
  // dim3 gridSize((nx+blockSize.x-1)/blockSize.x,(ny+blockSize.y-1)/blockSize.y,1);

  dim3 blockSize(32);
  dim3 gridSize(nx/32,ny);

  macroscopic_kernel<<<gridSize,blockSize>>>(params,v,fin_d,rho_d,ux_d,uy_d); 
  // macroscopic_kernel(params, 
  //                    v,
  //                    fin_d,
  //                    rho_d,
  //                    ux_d,
  //                    uy_d);

} // macroscopic

// ======================================================
// ======================================================
void equilibrium(const LBMParams& params, 
                 const velocity_array_t v,
                 const weights_t t,
                 const real_t* rho_d,
                 const real_t* ux_d,
                 const real_t* uy_d,
                 real_t* feq_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  // TODO : call kernel
  // unsigned int threadsPerBlockX=32;
  // unsigned int threadsPerBlockY=32;
  // dim3 blockSize(threadsPerBlockX,threadsPerBlockY,1);
  // dim3 gridSize((nx+blockSize.x-1)/blockSize.x,(ny+blockSize.y-1)/blockSize.y,1);

  dim3 blockSize(32);
  dim3 gridSize(nx/32,ny);
  equilibrium_kernel<<<gridSize,blockSize>>>(params,v,t,rho_d,ux_d,uy_d,feq_d); 
  // equilibrium_kernel(params, 
  //             v,
  //             t,
  //             rho_d,
  //             ux_d,
  //             uy_d,
  //             feq_d);

} // equilibrium

// ======================================================
// ======================================================
void init_obstacle_mask(const LBMParams& params, 
                        int* obstacle, 
                        int* obstacle_d)
{

  const int nx = params.nx;
  const int ny = params.ny;
  // std::cout << nx << std::endl;

  const real_t cx = params.cx;
  const real_t cy = params.cy;
  // std::cout << cx << std::endl;

  const real_t r = params.r;
  // std::cout << r << std::endl;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      real_t x = 1.0*i;
      real_t y = 1.0*j;

      obstacle[index] = (x-cx)*(x-cx) + (y-cy)*(y-cy) < r*r ? 1 : 0;
      // std::cout << obstacle[index] << std::endl;

    } // end for i
  } // end for j

  // TODO copy host to device
  CUDA_API_CHECK( cudaMemcpy( obstacle_d, obstacle, nx*ny * sizeof(int),
                              cudaMemcpyHostToDevice ) );
  // juste ce qu'il y a dans la signature ------------------------------------------------------------------------

  // CUDA_API_CHECK( cudaMemcpy( fin, fin_d, nx*ny*npop * sizeof(real_t),
  //                             cudaMemcpyDeviceToHost ) );
  // CUDA_API_CHECK( cudaMemcpy( fout, fout_d, nx*ny*npop * sizeof(real_t),
  //                             cudaMemcpyDeviceToHost ) );
  // CUDA_API_CHECK( cudaMemcpy( feq, feq_d, nx*ny*npop * sizeof(real_t),
  //                             cudaMemcpyDeviceToHost ) );

  // CUDA_API_CHECK( cudaMemcpy( rho, rho_d, nx*ny * sizeof(real_t),
  //                             cudaMemcpyDeviceToHost ) );
  // CUDA_API_CHECK( cudaMemcpy( ux, ux_d, nx*ny * sizeof(real_t),
  //                             cudaMemcpyDeviceToHost ) );
  // CUDA_API_CHECK( cudaMemcpy( uy, uy_d, nx*ny * sizeof(real_t),
  //                             cudaMemcpyDeviceToHost ) );

  // CUDA_API_CHECK( cudaMemcpy( obstacle, obstacle_d, nx*ny * sizeof(int),
  //                             cudaMemcpyDeviceToHost ) );

} // init_obstacle_mask

// ======================================================
// ======================================================
__host__ __device__
real_t compute_vel(int dir, int i, int j, real_t uLB, real_t ly)
{

  // flow is along X axis
  // X component is non-zero
  // Y component is always zero

  return (1-dir) * uLB * (1 + 1e-4 * sin(j/ly*2*M_PI));

} // compute_vel

// ======================================================
// ======================================================
void initialize_macroscopic_variables(const LBMParams& params, 
                                      real_t* rho, real_t* rho_d,
                                      real_t* ux, real_t* ux_d,
                                      real_t* uy, real_t* uy_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      rho[index] = 1.0;
      ux[index]  = compute_vel(0, i, j, params.uLB, params.ly);
      uy[index]  = compute_vel(1, i, j, params.uLB, params.ly);

    } // end for i
  } // end for j

  // TODO : copy host to device
  CUDA_API_CHECK( cudaMemcpy( rho_d, rho, nx*ny * sizeof(real_t),
                              cudaMemcpyHostToDevice ) );
  CUDA_API_CHECK( cudaMemcpy( ux_d, ux, nx*ny * sizeof(real_t),
                              cudaMemcpyHostToDevice ) );
  CUDA_API_CHECK( cudaMemcpy( uy_d, uy, nx*ny * sizeof(real_t),
                              cudaMemcpyHostToDevice ) );

} // initialize_macroscopic_variables

// ======================================================
// ======================================================
void border_outflow(const LBMParams& params, real_t* fin_d)
{

  // const int nx = params.nx;
  const int ny = params.ny;
  // rajouter nxny ?

  // TODO : call kernel
  // unsigned int threadsPerBlockX=32;
  // unsigned int threadsPerBlockY=32;
  // dim3 blockSize(threadsPerBlockX,threadsPerBlockY,1);
  // dim3 gridSize((nx+blockSize.x-1)/blockSize.x,(ny+blockSize.y-1)/blockSize.y,1);

  dim3 blockSize(32,1,1);
  dim3 gridSize(ny/32,1,1);
  border_outflow_kernel<<<gridSize,blockSize>>>(params,fin_d);
  // border_outflow_kernel(params, fin_d);

  // CUDA_API_CHECK( cudaMemcpy( fin_d, fin, nx*ny * sizeof(real_t),
  //                             cudaMemcpyHostToDevice ) );

} // border_outflow

// ======================================================
// ======================================================
void border_inflow(const LBMParams& params, const real_t* fin_d, 
                   real_t* rho_d, real_t* ux_d, real_t* uy_d)
{

  // const int nx = params.nx;
  const int ny = params.ny;
  // rajouter nxny ?

  // TODO : call kernel
  // unsigned int threadsPerBlockX=32;
  // unsigned int threadsPerBlockY=32;
  // dim3 blockSize(threadsPerBlockX,threadsPerBlockY,1);
  // dim3 gridSize((nx+blockSize.x-1)/blockSize.x,(ny+blockSize.y-1)/blockSize.y,1);

  dim3 blockSize(32);
  dim3 gridSize(ny/32);
  border_inflow_kernel<<<gridSize,blockSize>>>(params,fin_d,rho_d,ux_d,uy_d);
  // border_inflow_kernel(params, fin_d, 
  //                      rho_d, ux_d, uy_d);

} // border_inflow

// ======================================================
// ======================================================
void update_fin_inflow(const LBMParams& params, const real_t* feq_d, 
                       real_t* fin_d)
{

  // const int nx = params.nx;
  const int ny = params.ny;
  // rajouter nxny ?

  // TODO : call kernel
  // unsigned int threadsPerBlockX=32;
  // unsigned int threadsPerBlockY=32;
  // dim3 blockSize(threadsPerBlockX,threadsPerBlockY,1);
  // dim3 gridSize((nx+blockSize.x-1)/blockSize.x,(ny+blockSize.y-1)/blockSize.y,1);

  dim3 blockSize(32);
  dim3 gridSize(ny/32);
  update_fin_inflow_kernel<<<gridSize,blockSize>>>(params,feq_d,fin_d);
  // update_fin_inflow_kernel(params, feq_d, 
  //                          fin_d);

} // update_fin_inflow
  
// ======================================================
// ======================================================
void compute_collision(const LBMParams& params, 
                       const real_t* fin_d,
                       const real_t* feq_d,
                       real_t* fout_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  // TODO : call kernel
  // unsigned int threadsPerBlockX=32;
  // unsigned int threadsPerBlockY=32;
  // dim3 blockSize(threadsPerBlockX,threadsPerBlockY,1);
  // dim3 gridSize((nx+blockSize.x-1)/blockSize.x,(ny+blockSize.y-1)/blockSize.y,1);

  dim3 blockSize(32);
  dim3 gridSize(nx/32,ny);
  compute_collision_kernel<<<gridSize,blockSize>>>(params,fin_d,feq_d,fout_d);
  // compute_collision_kernel(params, 
  //                          fin_d,
  //                          feq_d,
  //                          fout_d);

} // compute_collision

// ======================================================
// ======================================================
void update_obstacle(const LBMParams &params, 
                     const real_t* fin_d,
                     const int* obstacle_d, 
                     real_t* fout_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  // TODO : call kernel
  // unsigned int threadsPerBlockX=32;
  // unsigned int threadsPerBlockY=32;
  // dim3 blockSize(threadsPerBlockX,threadsPerBlockY,1);
  // dim3 gridSize((nx+blockSize.x-1)/blockSize.x,(ny+blockSize.y-1)/blockSize.y,1);

  dim3 blockSize(32);
  dim3 gridSize(nx/32,ny);
  update_obstacle_kernel<<<gridSize,blockSize>>>(params,fin_d,obstacle_d,fout_d);
  // update_obstacle_kernel(&params, 
  //                        fin_d,
  //                        obstacle_d, 
  //                        fout_d);

} // update_obstacle

// ======================================================
// ======================================================
void streaming(const LBMParams& params,
               const velocity_array_t v,
               const real_t* fout_d,
               real_t* fin_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  // TODO : call kernel
  // unsigned int threadsPerBlockX=32;
  // unsigned int threadsPerBlockY=32;
  // dim3 blockSize(threadsPerBlockX,threadsPerBlockY,1);
  // dim3 gridSize((nx+blockSize.x-1)/blockSize.x,(ny+blockSize.y-1)/blockSize.y,1);

  dim3 blockSize(32);
  dim3 gridSize(nx/32,ny);
  streaming_kernel<<<gridSize,blockSize>>>(params,v,fout_d,fin_d);
  // streaming_kernel(params,
  //                  v,
  //                  fout_d,
  //                  fin_d);

} // streaming
