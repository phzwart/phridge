#pragma once

#include <cstdint>
#include <cuda_fp16.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace phridge_stamp_cuda {

constexpr int kMaxTerms = 8;
constexpr double kEightPi2 = 8.0 * M_PI * M_PI;
constexpr double kTwoPi = 2.0 * M_PI;
constexpr double kNorm0 = 0.06349363593424098;

template <typename T>
struct Acc {
  using type = T;
};

template <>
struct Acc<__half> {
  using type = float;
};

__device__ inline int wrap(int i, int n) {
  int r = i % n;
  return r < 0 ? r + n : r;
}

__device__ inline int iround(double x) { return __double2int_rn(x); }
__device__ inline int iround(float x) { return __float2int_rn(x); }

template <typename T>
__device__ inline void atomic_add_t(T* target, T value) {
  atomicAdd(target, value);
}

template <>
__device__ inline void atomic_add_t<double>(double* target, double value) {
#if __CUDA_ARCH__ >= 600
  atomicAdd(target, value);
#else
  unsigned long long* ip = reinterpret_cast<unsigned long long*>(target);
  unsigned long long old = *ip;
  unsigned long long assumed;
  do {
    assumed = old;
    old = atomicCAS(ip, assumed, __double_as_longlong(value + __longlong_as_double(assumed)));
  } while (assumed != old);
#endif
}

template <typename A>
__device__ inline A exp_lookup(const A* table, int ntab, A x) {
  A xs = x * A(-100);
  if (xs < A(0)) {
    return exp(x);
  }
  int i = static_cast<int>(static_cast<float>(xs) + 0.5f);
  if (i >= ntab) {
    return exp(x);
  }
  return table[i];
}

template <typename A>
__device__ inline A iso_gauss(A r2, A s2) {
  return pow(A(kTwoPi) * s2, A(-1.5)) * exp(A(-0.5) * r2 / s2);
}

template <typename A>
__device__ inline bool inv_det(
    A m00,
    A m01,
    A m02,
    A m11,
    A m12,
    A m22,
    A& p00,
    A& p01,
    A& p02,
    A& p11,
    A& p12,
    A& p22,
    A& det) {
  A c00 = m11 * m22 - m12 * m12;
  A c01 = m02 * m12 - m01 * m22;
  A c02 = m01 * m12 - m02 * m11;
  A c11 = m00 * m22 - m02 * m02;
  A c12 = m01 * m02 - m00 * m12;
  A c22 = m00 * m11 - m01 * m01;
  det = m00 * c00 + m01 * c01 + m02 * c02;
  if (det == A(0)) {
    return false;
  }
  p00 = c00 / det;
  p01 = c01 / det;
  p02 = c02 / det;
  p11 = c11 / det;
  p12 = c12 / det;
  p22 = c22 / det;
  return true;
}

template <typename T>
struct StampIn {
  const T* xf;
  const T* u_iso;
  const T* u_cart;
  const T* a;
  const T* b;
  const T* c;
  const T* w;
  const T* fp;
  const T* fdp;
  const T* r_cut;
  const std::uint8_t* aniso;
  const T* o;
  const typename Acc<T>::type* exp_table;
  int n_exp;
  int k_max;
  int n0, n1, n2;
  int ntab;
  T fs0, fs1, fs2;
  T u_extra;
  int has_imag;
};

template <typename T>
__global__ void stamp_forward_kernel(T* grid_re, T* grid_im, StampIn<T> in) {
  using A = typename Acc<T>::type;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= in.n_exp) {
    return;
  }
  const int n0 = in.n0, n1 = in.n1, n2 = in.n2;
  const A nf0 = static_cast<A>(n0);
  const A nf1 = static_cast<A>(n1);
  const A nf2 = static_cast<A>(n2);
  const A o00 = A(in.o[0]), o01 = A(in.o[1]), o02 = A(in.o[2]);
  const A o10 = A(in.o[3]), o11 = A(in.o[4]), o12 = A(in.o[5]);
  const A o20 = A(in.o[6]), o21 = A(in.o[7]), o22 = A(in.o[8]);
  const int k_max = in.k_max;
  const A x0 = A(in.xf[i * 3 + 0]);
  const A x1 = A(in.xf[i * 3 + 1]);
  const A x2 = A(in.xf[i * 3 + 2]);
  const A rc = A(in.r_cut[i]);
  const A rcut2 = rc * rc;
  const int h0 = static_cast<int>(ceil(static_cast<float>(rc * A(in.fs0))));
  const int h1 = static_cast<int>(ceil(static_cast<float>(rc * A(in.fs1))));
  const int h2 = static_cast<int>(ceil(static_cast<float>(rc * A(in.fs2))));
  const int g0 = iround(x0 * nf0);
  const int g1 = iround(x1 * nf1);
  const int g2 = iround(x2 * nf2);
  const double wi = in.w[i];
  const double cfp = in.c[i] + in.fp[i];
  const double fd = in.fdp[i];
  if (!in.aniso[i]) {
    const double s2b = in.u_iso[i] + in.u_extra;
    const int nterm = k_max + 1;
    double as_real[kMaxTerms];
    double bs_real[kMaxTerms];
    for (int kk = 0; kk < k_max; ++kk) {
      const double s2 = s2b + in.b[i * k_max + kk] / kEightPi2;
      as_real[kk] = wi * in.a[i * k_max + kk] * pow(kTwoPi * s2, -1.5);
      bs_real[kk] = -0.5 / s2;
    }
    as_real[k_max] = wi * cfp * pow(kTwoPi * s2b, -1.5);
    bs_real[k_max] = -0.5 / s2b;
    const double as_im = in.has_imag ? wi * fd * pow(kTwoPi * s2b, -1.5) : 0.0;
    for (int di = -h0; di <= h0; ++di) {
      const int gi0 = wrap(g0 + di, n0);
      const double rf0 = (g0 + di) / nf0 - x0;
      const double t0x = o00 * rf0, t0y = o10 * rf0, t0z = o20 * rf0;
      const int row0 = gi0 * n1;
      for (int dj = -h1; dj <= h1; ++dj) {
        const int gi1 = wrap(g1 + dj, n1);
        const double rf1 = (g1 + dj) / nf1 - x1;
        const double t1x = o01 * rf1, t1y = o11 * rf1, t1z = o21 * rf1;
        const int row01 = (row0 + gi1) * n2;
        for (int dk = -h2; dk <= h2; ++dk) {
          const int gi2 = wrap(g2 + dk, n2);
          const double rf2 = (g2 + dk) / nf2 - x2;
          const double rcx = t0x + t1x + o02 * rf2;
          const double rcy = t0y + t1y + o12 * rf2;
          const double rcz = t0z + t1z + o22 * rf2;
          const double r2 = rcx * rcx + rcy * rcy + rcz * rcz;
          if (r2 > rcut2) {
            continue;
          }
          double contr = 0.0;
          for (int kk = 0; kk < nterm; ++kk) {
            contr += as_real[kk] * exp_lookup(in.exp_table, in.ntab, bs_real[kk] * r2);
          }
          const int flat = row01 + gi2;
          atomic_add_t(grid_re + flat, T(contr));
          if (in.has_imag) {
            atomic_add_t(grid_im + flat, T(as_im * exp_lookup(in.exp_table, in.ntab, A(bs_real[k_max] * r2))));
          }
        }
      }
    }
  } else {
    const T* uc = in.u_cart + i * 9;
    const double u00 = uc[0], u01 = uc[1], u02 = uc[2];
    const double u11 = uc[4], u12 = uc[5], u22 = uc[8];
    for (int di = -h0; di <= h0; ++di) {
      const int gi0 = wrap(g0 + di, n0);
      const double rf0 = (g0 + di) / nf0 - x0;
      const double t0x = o00 * rf0, t0y = o10 * rf0, t0z = o20 * rf0;
      const int row0 = gi0 * n1;
      for (int dj = -h1; dj <= h1; ++dj) {
        const int gi1 = wrap(g1 + dj, n1);
        const double rf1 = (g1 + dj) / nf1 - x1;
        const double t1x = o01 * rf1, t1y = o11 * rf1, t1z = o21 * rf1;
        const int row01 = (row0 + gi1) * n2;
        for (int dk = -h2; dk <= h2; ++dk) {
          const int gi2 = wrap(g2 + dk, n2);
          const double rf2 = (g2 + dk) / nf2 - x2;
          const double rcx = t0x + t1x + o02 * rf2;
          const double rcy = t0y + t1y + o12 * rf2;
          const double rcz = t0z + t1z + o22 * rf2;
          const double r2 = rcx * rcx + rcy * rcy + rcz * rcz;
          if (r2 > rcut2) {
            continue;
          }
          double p00, p01, p02, p11, p12, p22, det;
          if (!inv_det(u00 + in.u_extra, u01, u02, u11 + in.u_extra, u12, u22 + in.u_extra, p00, p01, p02, p11, p12, p22, det)) {
            continue;
          }
          const double q = p00 * rcx * rcx + p11 * rcy * rcy + p22 * rcz * rcz +
                           2.0 * (p01 * rcx * rcy + p02 * rcx * rcz + p12 * rcy * rcz);
          const double base = kNorm0 * pow(det, -0.5) * exp_lookup(in.exp_table, in.ntab, -0.5 * q);
          double terms = 0.0;
          for (int kk = 0; kk < k_max; ++kk) {
            const double extra = in.b[i * k_max + kk] / kEightPi2 + in.u_extra;
            if (!inv_det(u00 + extra, u01, u02, u11 + extra, u12, u22 + extra, p00, p01, p02, p11, p12, p22, det)) {
              continue;
            }
            const double qk = p00 * rcx * rcx + p11 * rcy * rcy + p22 * rcz * rcz +
                              2.0 * (p01 * rcx * rcy + p02 * rcx * rcz + p12 * rcy * rcz);
            terms += in.a[i * k_max + kk] * (kNorm0 * pow(det, -0.5) * exp_lookup(in.exp_table, in.ntab, -0.5 * qk));
          }
          const int flat = row01 + gi2;
          atomic_add_t(grid_re + flat, T((terms + cfp * base) * wi));
          if (in.has_imag) {
            atomic_add_t(grid_im + flat, T(fd * base * wi));
          }
        }
      }
    }
  }
}

template <typename T>
struct VjpOut {
  T* gx;
  T* gu_iso;
  T* gucart;
  T* gw;
  T* gfp;
  T* gfdp;
};

template <typename T>
__global__ void stamp_vjp_kernel(const T* gre, const T* gim, StampIn<T> in, VjpOut<T> out) {
  using A = typename Acc<T>::type;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= in.n_exp) {
    return;
  }
  const int n0 = in.n0, n1 = in.n1, n2 = in.n2;
  const double nf0 = static_cast<double>(n0);
  const double nf1 = static_cast<double>(n1);
  const double nf2 = static_cast<double>(n2);
  const double o00 = in.o[0], o01 = in.o[1], o02 = in.o[2];
  const double o10 = in.o[3], o11 = in.o[4], o12 = in.o[5];
  const double o20 = in.o[6], o21 = in.o[7], o22 = in.o[8];
  const int k_max = in.k_max;
  const double x0 = in.xf[i * 3 + 0];
  const double x1 = in.xf[i * 3 + 1];
  const double x2 = in.xf[i * 3 + 2];
  const double rc = in.r_cut[i];
  const double rcut2 = rc * rc;
  const int h0 = static_cast<int>(ceil(rc * in.fs0));
  const int h1 = static_cast<int>(ceil(rc * in.fs1));
  const int h2 = static_cast<int>(ceil(rc * in.fs2));
  const int g0 = iround(x0 * nf0);
  const int g1 = iround(x1 * nf1);
  const int g2 = iround(x2 * nf2);
  const double wi = in.w[i];
  const double cfp = in.c[i] + in.fp[i];
  const double fd = in.fdp[i];
  double acc_x0 = 0, acc_x1 = 0, acc_x2 = 0, acc_u = 0;
  double acc_uc00 = 0, acc_uc01 = 0, acc_uc02 = 0, acc_uc11 = 0, acc_uc12 = 0, acc_uc22 = 0;
  double acc_w = 0, acc_fp = 0, acc_fdp = 0;
  if (!in.aniso[i]) {
    const double s2b = in.u_iso[i] + in.u_extra;
    for (int di = -h0; di <= h0; ++di) {
      const int gi0 = g0 + di;
      const double rf0 = gi0 / nf0 - x0;
      for (int dj = -h1; dj <= h1; ++dj) {
        const int gi1 = g1 + dj;
        const double rf1 = gi1 / nf1 - x1;
        for (int dk = -h2; dk <= h2; ++dk) {
          const int gi2 = g2 + dk;
          const double rf2 = gi2 / nf2 - x2;
          const double rcx = o00 * rf0 + o01 * rf1 + o02 * rf2;
          const double rcy = o10 * rf0 + o11 * rf1 + o12 * rf2;
          const double rcz = o20 * rf0 + o21 * rf1 + o22 * rf2;
          const double r2 = rcx * rcx + rcy * rcy + rcz * rcz;
          if (r2 > rcut2) {
            continue;
          }
          const double base = iso_gauss(r2, s2b);
          double terms = 0, d_terms_dr2 = 0, d_terms_ds2 = 0;
          for (int kk = 0; kk < k_max; ++kk) {
            const double s2k = s2b + in.b[i * k_max + kk] / kEightPi2;
            const double gk = iso_gauss(r2, s2k);
            const double ak = in.a[i * k_max + kk];
            terms += ak * gk;
            d_terms_dr2 += ak * gk * (-0.5 / s2k);
            d_terms_ds2 += ak * gk * (0.5 * r2 / (s2k * s2k) - 1.5 / s2k);
          }
          const double d_base_dr2 = base * (-0.5 / s2b);
          const double d_base_ds2 = base * (0.5 * r2 / (s2b * s2b) - 1.5 / s2b);
          const int flat = (wrap(gi0, n0) * n1 + wrap(gi1, n1)) * n2 + wrap(gi2, n2);
          const double er = gre[flat];
          const double ei = in.has_imag ? gim[flat] : 0.0;
          const double rho_re = terms + cfp * base;
          acc_w += er * rho_re + ei * (fd * base);
          acc_fp += er * wi * base;
          acc_fdp += ei * wi * base;
          const double dL_dr2 = er * wi * (d_terms_dr2 + cfp * d_base_dr2) + ei * wi * fd * d_base_dr2;
          const double ot0 = o00 * rcx + o10 * rcy + o20 * rcz;
          const double ot1 = o01 * rcx + o11 * rcy + o21 * rcz;
          const double ot2 = o02 * rcx + o12 * rcy + o22 * rcz;
          acc_x0 += dL_dr2 * (-2.0 * ot0);
          acc_x1 += dL_dr2 * (-2.0 * ot1);
          acc_x2 += dL_dr2 * (-2.0 * ot2);
          acc_u += er * wi * (d_terms_ds2 + cfp * d_base_ds2) + ei * wi * fd * d_base_ds2;
        }
      }
    }
  } else {
    const T* uc = in.u_cart + i * 9;
    const double u00 = uc[0], u01 = uc[1], u02 = uc[2];
    const double u11 = uc[4], u12 = uc[5], u22 = uc[8];
    for (int di = -h0; di <= h0; ++di) {
      const int gi0 = g0 + di;
      const double rf0 = gi0 / nf0 - x0;
      for (int dj = -h1; dj <= h1; ++dj) {
        const int gi1 = g1 + dj;
        const double rf1 = gi1 / nf1 - x1;
        for (int dk = -h2; dk <= h2; ++dk) {
          const int gi2 = g2 + dk;
          const double rf2 = gi2 / nf2 - x2;
          const double rcx = o00 * rf0 + o01 * rf1 + o02 * rf2;
          const double rcy = o10 * rf0 + o11 * rf1 + o12 * rf2;
          const double rcz = o20 * rf0 + o21 * rf1 + o22 * rf2;
          const double r2 = rcx * rcx + rcy * rcy + rcz * rcz;
          if (r2 > rcut2) {
            continue;
          }
          const int flat = (wrap(gi0, n0) * n1 + wrap(gi1, n1)) * n2 + wrap(gi2, n2);
          const double er = gre[flat];
          const double ei = in.has_imag ? gim[flat] : 0.0;
          double p00, p01, p02, p11, p12, p22, det;
          if (!inv_det(u00 + in.u_extra, u01, u02, u11 + in.u_extra, u12, u22 + in.u_extra, p00, p01, p02, p11, p12, p22, det)) {
            continue;
          }
          const double q = p00 * rcx * rcx + p11 * rcy * rcy + p22 * rcz * rcz +
                           2.0 * (p01 * rcx * rcy + p02 * rcx * rcz + p12 * rcy * rcz);
          const double base = kNorm0 * pow(det, -0.5) * exp(-0.5 * q);
          const double pr0 = p00 * rcx + p01 * rcy + p02 * rcz;
          const double pr1 = p01 * rcx + p11 * rcy + p12 * rcz;
          const double pr2 = p02 * rcx + p12 * rcy + p22 * rcz;
          double terms = 0, dx0 = 0, dx1 = 0, dx2 = 0;
          double duc00 = 0, duc01 = 0, duc02 = 0, duc11 = 0, duc12 = 0, duc22 = 0;
          const double scale_base = er * wi * cfp + ei * wi * fd;
          dx0 += scale_base * base * (o00 * pr0 + o10 * pr1 + o20 * pr2);
          dx1 += scale_base * base * (o01 * pr0 + o11 * pr1 + o21 * pr2);
          dx2 += scale_base * base * (o02 * pr0 + o12 * pr1 + o22 * pr2);
          const double sb = scale_base * base;
          duc00 += sb * (-0.5 * p00 + 0.5 * pr0 * pr0);
          duc01 += sb * (-0.5 * p01 + 0.5 * pr0 * pr1);
          duc02 += sb * (-0.5 * p02 + 0.5 * pr0 * pr2);
          duc11 += sb * (-0.5 * p11 + 0.5 * pr1 * pr1);
          duc12 += sb * (-0.5 * p12 + 0.5 * pr1 * pr2);
          duc22 += sb * (-0.5 * p22 + 0.5 * pr2 * pr2);
          for (int kk = 0; kk < k_max; ++kk) {
            const double extra = in.b[i * k_max + kk] / kEightPi2 + in.u_extra;
            if (!inv_det(u00 + extra, u01, u02, u11 + extra, u12, u22 + extra, p00, p01, p02, p11, p12, p22, det)) {
              continue;
            }
            const double qk = p00 * rcx * rcx + p11 * rcy * rcy + p22 * rcz * rcz +
                              2.0 * (p01 * rcx * rcy + p02 * rcx * rcz + p12 * rcy * rcz);
            const double gk = kNorm0 * pow(det, -0.5) * exp(-0.5 * qk);
            const double prk0 = p00 * rcx + p01 * rcy + p02 * rcz;
            const double prk1 = p01 * rcx + p11 * rcy + p12 * rcz;
            const double prk2 = p02 * rcx + p12 * rcy + p22 * rcz;
            terms += in.a[i * k_max + kk] * gk;
            const double sk = er * wi * in.a[i * k_max + kk];
            dx0 += sk * gk * (o00 * prk0 + o10 * prk1 + o20 * prk2);
            dx1 += sk * gk * (o01 * prk0 + o11 * prk1 + o21 * prk2);
            dx2 += sk * gk * (o02 * prk0 + o12 * prk1 + o22 * prk2);
            const double sg = sk * gk;
            duc00 += sg * (-0.5 * p00 + 0.5 * prk0 * prk0);
            duc01 += sg * (-0.5 * p01 + 0.5 * prk0 * prk1);
            duc02 += sg * (-0.5 * p02 + 0.5 * prk0 * prk2);
            duc11 += sg * (-0.5 * p11 + 0.5 * prk1 * prk1);
            duc12 += sg * (-0.5 * p12 + 0.5 * prk1 * prk2);
            duc22 += sg * (-0.5 * p22 + 0.5 * prk2 * prk2);
          }
          acc_w += er * (terms + cfp * base) + ei * (fd * base);
          acc_fp += er * wi * base;
          acc_fdp += ei * wi * base;
          acc_x0 += dx0;
          acc_x1 += dx1;
          acc_x2 += dx2;
          acc_uc00 += duc00;
          acc_uc01 += duc01;
          acc_uc02 += duc02;
          acc_uc11 += duc11;
          acc_uc12 += duc12;
          acc_uc22 += duc22;
        }
      }
    }
  }
  out.gx[i * 3 + 0] = T(acc_x0);
  out.gx[i * 3 + 1] = T(acc_x1);
  out.gx[i * 3 + 2] = T(acc_x2);
  out.gu_iso[i] = T(acc_u);
  out.gucart[i * 9 + 0] = T(acc_uc00);
  out.gucart[i * 9 + 1] = T(acc_uc01);
  out.gucart[i * 9 + 2] = T(acc_uc02);
  out.gucart[i * 9 + 3] = T(acc_uc01);
  out.gucart[i * 9 + 4] = T(acc_uc11);
  out.gucart[i * 9 + 5] = T(acc_uc12);
  out.gucart[i * 9 + 6] = T(acc_uc02);
  out.gucart[i * 9 + 7] = T(acc_uc12);
  out.gucart[i * 9 + 8] = T(acc_uc22);
  out.gw[i] = T(acc_w);
  out.gfp[i] = T(acc_fp);
  out.gfdp[i] = T(acc_fdp);
}

template <typename T>
inline void launch_forward(T* grid_re, T* grid_im, const StampIn<T>& in) {
  if (in.n_exp <= 0) {
    return;
  }
  const int threads = 64;
  const int blocks = (in.n_exp + threads - 1) / threads;
  stamp_forward_kernel<<<blocks, threads>>>(grid_re, grid_im, in);
}

template <typename T>
inline void launch_vjp(const T* gre, const T* gim, const StampIn<T>& in, const VjpOut<T>& out) {
  if (in.n_exp <= 0) {
    return;
  }
  const int threads = 64;
  const int blocks = (in.n_exp + threads - 1) / threads;
  stamp_vjp_kernel<<<blocks, threads>>>(gre, gim, in, out);
}

}  // namespace phridge_stamp_cuda
