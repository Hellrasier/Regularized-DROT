#include <cuda_runtime.h>
#include <stdexcept>
#include "drot_qr.hpp"

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess) \
throw std::runtime_error(cudaGetErrorString(e)); }while(0)

template <typename T>
void qr_run_into_device_bridge(
    const T* c_dev, const T* p_dev, const T* q_dev,
    int n_rows, int n_cols,
    T step_size, T r_weight,
    int max_iters, T eps,
    bool use_warmup_init,
    T* x_out_dev,                  // may be nullptr
    float* total_ms, float* prep_ms,
    int* n_iter, T* objective,
    cudaStream_t stream = nullptr)
{
    quadratic_regularizer_drot_wrapper<T>(
        c_dev, p_dev, q_dev,
        n_rows, n_cols,
        step_size, r_weight,
        max_iters, eps,
        use_warmup_init,
        x_out_dev,
        total_ms, prep_ms, n_iter, objective,
        stream
    );
}

// explicit instantiations
template void qr_run_into_device_bridge<float>(const float*, const float*, const float*,
  int,int,float,float,int,float,bool,float*,float*,int*,float*,cudaStream_t);
template void qr_run_into_device_bridge<double>(const double*, const double*, const double*,
  int,int,double,double,int,double,bool,double*,float*,int*,double*,cudaStream_t);