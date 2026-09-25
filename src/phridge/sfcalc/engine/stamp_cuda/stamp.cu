#include <cuda_fp16.h>
#include <stdexcept>

#include <torch/extension.h>

#include "kernels.cuh"

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

template <typename T>
T* raw(torch::Tensor t) {
  return reinterpret_cast<T*>(t.data_ptr());
}

template <typename T, typename A>
phridge_stamp_cuda::StampIn<T> make_in(
    torch::Tensor xf,
    torch::Tensor u_iso,
    torch::Tensor u_cart,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor w,
    torch::Tensor fp,
    torch::Tensor fdp,
    torch::Tensor r_cut,
    torch::Tensor aniso,
    torch::Tensor o,
    torch::Tensor exp_table,
    int n0,
    int n1,
    int n2,
    double fs0,
    double fs1,
    double fs2,
    double u_extra,
    int has_imag) {
  CHECK_CUDA(xf);
  CHECK_CONTIG(xf);
  if (xf.dim() != 2 || xf.size(1) != 3) {
    throw std::runtime_error("xf must be (n,3)");
  }
  if (a.dim() != 2) {
    throw std::runtime_error("a must be (n,k)");
  }
  const int n_exp = static_cast<int>(xf.size(0));
  const int k_max = static_cast<int>(a.size(1));
  if (k_max >= phridge_stamp_cuda::kMaxTerms) {
    throw std::runtime_error("too many Gaussian terms");
  }
  phridge_stamp_cuda::StampIn<T> in{};
  in.xf = raw<T>(xf);
  in.u_iso = raw<T>(u_iso);
  in.u_cart = raw<T>(u_cart);
  in.a = raw<T>(a);
  in.b = raw<T>(b);
  in.c = raw<T>(c);
  in.w = raw<T>(w);
  in.fp = raw<T>(fp);
  in.fdp = raw<T>(fdp);
  in.r_cut = raw<T>(r_cut);
  in.aniso = aniso.data_ptr<std::uint8_t>();
  in.o = raw<T>(o);
  in.exp_table = raw<A>(exp_table);
  in.n_exp = n_exp;
  in.k_max = k_max;
  in.n0 = n0;
  in.n1 = n1;
  in.n2 = n2;
  in.ntab = static_cast<int>(exp_table.numel());
  in.fs0 = static_cast<T>(fs0);
  in.fs1 = static_cast<T>(fs1);
  in.fs2 = static_cast<T>(fs2);
  in.u_extra = static_cast<T>(u_extra);
  in.has_imag = has_imag;
  return in;
}

template <typename T, typename A>
void forward_t(
    torch::Tensor grid_re,
    torch::Tensor grid_im,
    torch::Tensor xf,
    torch::Tensor u_iso,
    torch::Tensor u_cart,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor w,
    torch::Tensor fp,
    torch::Tensor fdp,
    torch::Tensor r_cut,
    torch::Tensor aniso,
    torch::Tensor o,
    torch::Tensor exp_table,
    int64_t n0,
    int64_t n1,
    int64_t n2,
    double fs0,
    double fs1,
    double fs2,
    double u_extra,
    int64_t has_imag) {
  auto in = make_in<T, A>(
      xf, u_iso, u_cart, a, b, c, w, fp, fdp, r_cut, aniso, o, exp_table, static_cast<int>(n0),
      static_cast<int>(n1), static_cast<int>(n2), fs0, fs1, fs2, u_extra, static_cast<int>(has_imag));
  phridge_stamp_cuda::launch_forward(raw<T>(grid_re), raw<T>(grid_im), in);
}

template <typename T, typename A>
void vjp_t(
    torch::Tensor gre,
    torch::Tensor gim,
    torch::Tensor xf,
    torch::Tensor u_iso,
    torch::Tensor u_cart,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor w,
    torch::Tensor fp,
    torch::Tensor fdp,
    torch::Tensor r_cut,
    torch::Tensor aniso,
    torch::Tensor o,
    torch::Tensor exp_table,
    int64_t n0,
    int64_t n1,
    int64_t n2,
    double fs0,
    double fs1,
    double fs2,
    double u_extra,
    int64_t has_imag,
    torch::Tensor gx,
    torch::Tensor gu_iso,
    torch::Tensor gucart,
    torch::Tensor gw,
    torch::Tensor gfp,
    torch::Tensor gfdp) {
  auto in = make_in<T, A>(
      xf, u_iso, u_cart, a, b, c, w, fp, fdp, r_cut, aniso, o, exp_table, static_cast<int>(n0),
      static_cast<int>(n1), static_cast<int>(n2), fs0, fs1, fs2, u_extra, static_cast<int>(has_imag));
  phridge_stamp_cuda::VjpOut<T> out{};
  out.gx = raw<T>(gx);
  out.gu_iso = raw<T>(gu_iso);
  out.gucart = raw<T>(gucart);
  out.gw = raw<T>(gw);
  out.gfp = raw<T>(gfp);
  out.gfdp = raw<T>(gfdp);
  phridge_stamp_cuda::launch_vjp(raw<T>(gre), raw<T>(gim), in, out);
}

#define DISPATCH_PREC(fn)                                                                 \
  do {                                                                                    \
    auto st = grid_like.scalar_type();                                                    \
    if (st == torch::kFloat64) {                                                          \
      fn(double, double);                                                                 \
    } else if (st == torch::kFloat) {                                                     \
      fn(float, float);                                                                   \
    } else if (st == torch::kHalf) {                                                      \
      fn(__half, float);                                                                  \
    } else {                                                                              \
      throw std::runtime_error("stamp cuda dtype must be float64, float32, or float16"); \
    }                                                                                     \
  } while (0)

void stamp_forward(
    torch::Tensor grid_re,
    torch::Tensor grid_im,
    torch::Tensor xf,
    torch::Tensor u_iso,
    torch::Tensor u_cart,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor w,
    torch::Tensor fp,
    torch::Tensor fdp,
    torch::Tensor r_cut,
    torch::Tensor aniso,
    torch::Tensor o,
    torch::Tensor exp_table,
    int64_t n0,
    int64_t n1,
    int64_t n2,
    double fs0,
    double fs1,
    double fs2,
    double u_extra,
    int64_t has_imag) {
  CHECK_CUDA(grid_re);
  CHECK_CONTIG(grid_re);
  torch::Tensor grid_like = grid_re;
#define CALL_FWD(T, A) \
  forward_t<T, A>(grid_re, grid_im, xf, u_iso, u_cart, a, b, c, w, fp, fdp, r_cut, aniso, o, exp_table, n0, n1, n2, fs0, fs1, fs2, u_extra, has_imag)
  DISPATCH_PREC(CALL_FWD);
#undef CALL_FWD
}

void stamp_vjp(
    torch::Tensor gre,
    torch::Tensor gim,
    torch::Tensor xf,
    torch::Tensor u_iso,
    torch::Tensor u_cart,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor w,
    torch::Tensor fp,
    torch::Tensor fdp,
    torch::Tensor r_cut,
    torch::Tensor aniso,
    torch::Tensor o,
    torch::Tensor exp_table,
    int64_t n0,
    int64_t n1,
    int64_t n2,
    double fs0,
    double fs1,
    double fs2,
    double u_extra,
    int64_t has_imag,
    torch::Tensor gx,
    torch::Tensor gu_iso,
    torch::Tensor gucart,
    torch::Tensor gw,
    torch::Tensor gfp,
    torch::Tensor gfdp) {
  CHECK_CUDA(gre);
  CHECK_CONTIG(gre);
  torch::Tensor grid_like = gre;
#define CALL_VJP(T, A) \
  vjp_t<T, A>(gre, gim, xf, u_iso, u_cart, a, b, c, w, fp, fdp, r_cut, aniso, o, exp_table, n0, n1, n2, fs0, fs1, fs2, u_extra, has_imag, gx, gu_iso, gucart, gw, gfp, gfdp)
  DISPATCH_PREC(CALL_VJP);
#undef CALL_VJP
}

#undef CHECK_CUDA
#undef CHECK_CONTIG
#undef DISPATCH_PREC

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stamp_forward", &stamp_forward);
  m.def("stamp_vjp", &stamp_vjp);
}
