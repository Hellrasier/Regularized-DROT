// src/bindings.cpp
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

// bridge symbols:
template <typename T>
void qr_run_into_device(const T* c_dev, const T* p_dev, const T* q_dev,
    int n_rows, int n_cols,
    T step_size, T r_weight,
    int max_iters, T eps,
    bool use_warmup_init,
    T* x_out_dev,              // may be null
    float* total_ms, float* prep_ms,
    int* n_iter, T* objective);

template <typename T>
void launch_qr_uintptr(uintptr_t c_ptr, uintptr_t p_ptr, uintptr_t q_ptr,
                       int n_rows, int n_cols,
                       T step_size, T r_weight,
                       int max_iters, T eps,
                       bool use_warmup_init,
                       uintptr_t x_out_ptr,     // 0 means "don’t return the plan"
                       py::object metrics_out)
{
    const T* c = reinterpret_cast<const T*>(c_ptr);
    const T* p = reinterpret_cast<const T*>(p_ptr);
    const T* q = reinterpret_cast<const T*>(q_ptr);
    T* x_out   = x_out_ptr ? reinterpret_cast<T*>(x_out_ptr) : nullptr;

    float total_ms = 0.f, prep_ms = 0.f;
    int n_iter = 0;
    T objective = 0;

    qr_run_into_device<T>(c, p, q, n_rows, n_cols,
        step_size, r_weight, max_iters, eps,
        use_warmup_init, x_out,
        &total_ms, &prep_ms, &n_iter, &objective);

    if (!metrics_out.is_none()) {
        metrics_out.attr("__setitem__")("total_ms", total_ms);
        metrics_out.attr("__setitem__")("prep_ms",  prep_ms);
        metrics_out.attr("__setitem__")("n_iter",   n_iter);
        metrics_out.attr("__setitem__")("objective", objective);
    }
}

PYBIND11_MODULE(qr_cuda, m) {
    m.doc() = "Quadratic-regularized OT CUDA bindings (device-pointer API)";
    m.def("run_f32", &launch_qr_uintptr<float>,
          "Run float32 QR-OT on device arrays",
          py::arg("c_ptr"), py::arg("p_ptr"), py::arg("q_ptr"),
          py::arg("n_rows"), py::arg("n_cols"),
          py::arg("step_size"), py::arg("r_weight"),
          py::arg("max_iters"), py::arg("eps"),
          py::arg("use_warmup_init") = false,
          py::arg("x_out_ptr") = 0,
          py::arg("metrics") = py::none());
    m.def("run_f64", &launch_qr_uintptr<double>,
          "Run float64 QR-OT on device arrays",
          py::arg("c_ptr"), py::arg("p_ptr"), py::arg("q_ptr"),
          py::arg("n_rows"), py::arg("n_cols"),
          py::arg("step_size"), py::arg("r_weight"),
          py::arg("max_iters"), py::arg("eps"),
          py::arg("use_warmup_init") = false,
          py::arg("x_out_ptr") = 0,
          py::arg("metrics") = py::none());
}