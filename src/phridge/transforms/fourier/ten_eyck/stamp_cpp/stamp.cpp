#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

#include "kernels.h"

namespace py = pybind11;

using ArrU8 = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

template <typename T>
using Arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T, typename A>
static phridge_stamp::StampIn<T> make_in(
    Arr<T> xf,
    Arr<T> u_iso,
    Arr<T> u_cart,
    Arr<T> a,
    Arr<T> b,
    Arr<T> c,
    Arr<T> w,
    Arr<T> fp,
    Arr<T> fdp,
    Arr<T> r_cut,
    ArrU8 aniso,
    Arr<T> o,
    Arr<A> exp_table,
    int n0,
    int n1,
    int n2,
    double fs0,
    double fs1,
    double fs2,
    double u_extra,
    int has_imag) {
  auto xf_b = xf.request();
  auto a_b = a.request();
  if (xf_b.ndim != 2 || xf_b.shape[1] != 3) {
    throw std::runtime_error("xf must be (n,3)");
  }
  if (a_b.ndim != 2) {
    throw std::runtime_error("a must be (n,k)");
  }
  const int n_exp = static_cast<int>(xf_b.shape[0]);
  const int k_max = static_cast<int>(a_b.shape[1]);
  if (k_max >= phridge_stamp::kMaxTerms) {
    throw std::runtime_error("too many Gaussian terms");
  }
  auto et = exp_table.request();
  phridge_stamp::StampIn<T> in{};
  in.xf = static_cast<const T*>(xf_b.ptr);
  in.u_iso = static_cast<const T*>(u_iso.request().ptr);
  in.u_cart = static_cast<const T*>(u_cart.request().ptr);
  in.a = static_cast<const T*>(a_b.ptr);
  in.b = static_cast<const T*>(b.request().ptr);
  in.c = static_cast<const T*>(c.request().ptr);
  in.w = static_cast<const T*>(w.request().ptr);
  in.fp = static_cast<const T*>(fp.request().ptr);
  in.fdp = static_cast<const T*>(fdp.request().ptr);
  in.r_cut = static_cast<const T*>(r_cut.request().ptr);
  in.aniso = static_cast<const std::uint8_t*>(aniso.request().ptr);
  in.o = static_cast<const T*>(o.request().ptr);
  in.exp_table = static_cast<const A*>(et.ptr);
  in.n_exp = n_exp;
  in.k_max = k_max;
  in.n0 = n0;
  in.n1 = n1;
  in.n2 = n2;
  in.ntab = static_cast<int>(et.shape[0]);
  in.fs0 = static_cast<T>(fs0);
  in.fs1 = static_cast<T>(fs1);
  in.fs2 = static_cast<T>(fs2);
  in.u_extra = static_cast<T>(u_extra);
  in.has_imag = has_imag;
  return in;
}

template <typename T, typename A>
static void stamp_forward_py(
    Arr<T> grid_re,
    Arr<T> grid_im,
    Arr<T> xf,
    Arr<T> u_iso,
    Arr<T> u_cart,
    Arr<T> a,
    Arr<T> b,
    Arr<T> c,
    Arr<T> w,
    Arr<T> fp,
    Arr<T> fdp,
    Arr<T> r_cut,
    ArrU8 aniso,
    Arr<T> o,
    Arr<A> exp_table,
    int n0,
    int n1,
    int n2,
    double fs0,
    double fs1,
    double fs2,
    double u_extra,
    int has_imag) {
  auto in = make_in<T, A>(
      xf, u_iso, u_cart, a, b, c, w, fp, fdp, r_cut, aniso, o, exp_table, n0, n1, n2, fs0, fs1, fs2, u_extra, has_imag);
  phridge_stamp::stamp_forward(
      static_cast<T*>(grid_re.request().ptr), static_cast<T*>(grid_im.request().ptr), in);
}

template <typename T, typename A>
static void stamp_vjp_py(
    Arr<T> gre,
    Arr<T> gim,
    Arr<T> xf,
    Arr<T> u_iso,
    Arr<T> u_cart,
    Arr<T> a,
    Arr<T> b,
    Arr<T> c,
    Arr<T> w,
    Arr<T> fp,
    Arr<T> fdp,
    Arr<T> r_cut,
    ArrU8 aniso,
    Arr<T> o,
    Arr<A> exp_table,
    int n0,
    int n1,
    int n2,
    double fs0,
    double fs1,
    double fs2,
    double u_extra,
    int has_imag,
    Arr<T> gx,
    Arr<T> gu_iso,
    Arr<T> gucart,
    Arr<T> gw,
    Arr<T> gfp,
    Arr<T> gfdp) {
  auto in = make_in<T, A>(
      xf, u_iso, u_cart, a, b, c, w, fp, fdp, r_cut, aniso, o, exp_table, n0, n1, n2, fs0, fs1, fs2, u_extra, has_imag);
  phridge_stamp::VjpOut<T> out{};
  out.gx = static_cast<T*>(gx.request().ptr);
  out.gu_iso = static_cast<T*>(gu_iso.request().ptr);
  out.gucart = static_cast<T*>(gucart.request().ptr);
  out.gw = static_cast<T*>(gw.request().ptr);
  out.gfp = static_cast<T*>(gfp.request().ptr);
  out.gfdp = static_cast<T*>(gfdp.request().ptr);
  phridge_stamp::stamp_vjp(static_cast<const T*>(gre.request().ptr), static_cast<const T*>(gim.request().ptr), in, out);
}

#if PHRIDGE_HAS_F16
static phridge_f16* f16_mut(py::array& a) {
  if (a.itemsize() != 2) {
    throw std::runtime_error("float16 array required");
  }
  return reinterpret_cast<phridge_f16*>(a.mutable_data());
}

static const phridge_f16* f16_const(const py::array& a) {
  if (a.itemsize() != 2) {
    throw std::runtime_error("float16 array required");
  }
  return reinterpret_cast<const phridge_f16*>(a.data());
}

static phridge_stamp::StampIn<phridge_f16> make_in_f16(
    py::array xf,
    py::array u_iso,
    py::array u_cart,
    py::array a,
    py::array b,
    py::array c,
    py::array w,
    py::array fp,
    py::array fdp,
    py::array r_cut,
    ArrU8 aniso,
    py::array o,
    Arr<float> exp_table,
    int n0,
    int n1,
    int n2,
    double fs0,
    double fs1,
    double fs2,
    double u_extra,
    int has_imag) {
  if (xf.ndim() != 2 || xf.shape(1) != 3) {
    throw std::runtime_error("xf must be (n,3)");
  }
  if (a.ndim() != 2) {
    throw std::runtime_error("a must be (n,k)");
  }
  const int n_exp = static_cast<int>(xf.shape(0));
  const int k_max = static_cast<int>(a.shape(1));
  if (k_max >= phridge_stamp::kMaxTerms) {
    throw std::runtime_error("too many Gaussian terms");
  }
  auto et = exp_table.request();
  phridge_stamp::StampIn<phridge_f16> in{};
  in.xf = f16_const(xf);
  in.u_iso = f16_const(u_iso);
  in.u_cart = f16_const(u_cart);
  in.a = f16_const(a);
  in.b = f16_const(b);
  in.c = f16_const(c);
  in.w = f16_const(w);
  in.fp = f16_const(fp);
  in.fdp = f16_const(fdp);
  in.r_cut = f16_const(r_cut);
  in.aniso = static_cast<const std::uint8_t*>(aniso.request().ptr);
  in.o = f16_const(o);
  in.exp_table = static_cast<const float*>(et.ptr);
  in.n_exp = n_exp;
  in.k_max = k_max;
  in.n0 = n0;
  in.n1 = n1;
  in.n2 = n2;
  in.ntab = static_cast<int>(et.shape[0]);
  in.fs0 = static_cast<phridge_f16>(fs0);
  in.fs1 = static_cast<phridge_f16>(fs1);
  in.fs2 = static_cast<phridge_f16>(fs2);
  in.u_extra = static_cast<phridge_f16>(u_extra);
  in.has_imag = has_imag;
  return in;
}

static void stamp_forward_f16(
    py::array grid_re,
    py::array grid_im,
    py::array xf,
    py::array u_iso,
    py::array u_cart,
    py::array a,
    py::array b,
    py::array c,
    py::array w,
    py::array fp,
    py::array fdp,
    py::array r_cut,
    ArrU8 aniso,
    py::array o,
    Arr<float> exp_table,
    int n0,
    int n1,
    int n2,
    double fs0,
    double fs1,
    double fs2,
    double u_extra,
    int has_imag) {
  auto in = make_in_f16(
      xf, u_iso, u_cart, a, b, c, w, fp, fdp, r_cut, aniso, o, exp_table, n0, n1, n2, fs0, fs1, fs2, u_extra, has_imag);
  phridge_stamp::stamp_forward(f16_mut(grid_re), f16_mut(grid_im), in);
}

static void stamp_vjp_f16(
    py::array gre,
    py::array gim,
    py::array xf,
    py::array u_iso,
    py::array u_cart,
    py::array a,
    py::array b,
    py::array c,
    py::array w,
    py::array fp,
    py::array fdp,
    py::array r_cut,
    ArrU8 aniso,
    py::array o,
    Arr<float> exp_table,
    int n0,
    int n1,
    int n2,
    double fs0,
    double fs1,
    double fs2,
    double u_extra,
    int has_imag,
    py::array gx,
    py::array gu_iso,
    py::array gucart,
    py::array gw,
    py::array gfp,
    py::array gfdp) {
  auto in = make_in_f16(
      xf, u_iso, u_cart, a, b, c, w, fp, fdp, r_cut, aniso, o, exp_table, n0, n1, n2, fs0, fs1, fs2, u_extra, has_imag);
  phridge_stamp::VjpOut<phridge_f16> out{};
  out.gx = f16_mut(gx);
  out.gu_iso = f16_mut(gu_iso);
  out.gucart = f16_mut(gucart);
  out.gw = f16_mut(gw);
  out.gfp = f16_mut(gfp);
  out.gfdp = f16_mut(gfdp);
  phridge_stamp::stamp_vjp(f16_const(gre), f16_const(gim), in, out);
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stamp_forward", &stamp_forward_py<double, double>);
  m.def("stamp_vjp", &stamp_vjp_py<double, double>);
  m.def("stamp_forward_f32", &stamp_forward_py<float, float>);
  m.def("stamp_vjp_f32", &stamp_vjp_py<float, float>);
#if PHRIDGE_HAS_F16
  m.def("has_f16", []() { return true; });
  m.def("stamp_forward_f16", &stamp_forward_f16);
  m.def("stamp_vjp_f16", &stamp_vjp_f16);
#else
  m.def("has_f16", []() { return false; });
#endif
}
