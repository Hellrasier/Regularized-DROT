#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>
#include <cstdio>

#include "drot_qr.hpp"     // declares quadratic_regularizer_drot_wrapper<T>
#include "kernel_qr.hpp"
#include "param_qr.hpp"

#define CUDA_CHECK(expr) do {                                    \
  cudaError_t _e = (expr);                                       \
  if (_e != cudaSuccess) {                                       \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,\
            cudaGetErrorString(_e));                             \
    throw std::runtime_error(cudaGetErrorString(_e));            \
  }                                                              \
} while(0)

// This bridge calls your existing wrapper (which returns a host pointer)
// and, if an output device pointer is provided, copies the result to it.
template <typename T>
void qr_run_into_device(
    const T* c_dev, const T* p_dev, const T* q_dev,
    int n_rows, int n_cols,
    T step_size, T r_weight,
    int max_iters, T eps,
    bool use_warmup_init,
    T* x_out_dev,                // optional (device). If nullptr, we just free after timing.
    float* total_ms, float* prep_ms,
    int* n_iter, T* objective)
{
    // Call the existing function (allocs GPU temporaries, returns host copy if return_x=true)
    bool return_x = (x_out_dev == nullptr);        // if no device out, we still compute for metrics
    T* host_out = quadratic_regularizer_drot_wrapper<T>(
        c_dev, p_dev, q_dev,
        n_rows, n_cols,
        step_size, r_weight,
        max_iters, eps,
        total_ms, prep_ms,
        n_iter, objective,
        use_warmup_init,
        /*return_x=*/true   // we want the plan back (host pointer)
    );

    if (x_out_dev) {
        size_t bytes = static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols) * sizeof(T);
        CUDA_CHECK(cudaMemcpy(x_out_dev, host_out, bytes, cudaMemcpyHostToDevice));
    }
    // wrapper allocates host_out with malloc -> caller frees
    free(host_out);
}

// explicit instantiations so the symbols exist for the binder
template void qr_run_into_device<float>(const float*, const float*, const float*,
    int,int,float,float,int,float,bool,float*,float*,int*,float*);
template void qr_run_into_device<double>(const double*, const double*, const double*,
    int,int,double,double,int,double,bool,double*,float*,float*,int*,double*);