#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(__FLT16_MANT_DIG__) || defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
#define PHRIDGE_HAS_F16 1
using phridge_f16 = _Float16;
#else
#define PHRIDGE_HAS_F16 0
#endif

namespace phridge_stamp {

constexpr int kMaxTerms = 8;
constexpr double kEightPi2 = 8.0 * M_PI * M_PI;
constexpr double kTwoPi = 2.0 * M_PI;
constexpr double kNorm0 = 0.06349363593424098;  // (2π)^{-3/2}

template <typename T>
struct Acc {
  using type = T;
};

#if PHRIDGE_HAS_F16
template <>
struct Acc<phridge_f16> {
  using type = float;
};
#endif

inline int wrap(int i, int n) {
  int r = i % n;
  return r < 0 ? r + n : r;
}

template <typename A>
inline int iround(A x) {
  return static_cast<int>(std::lrint(static_cast<double>(x)));
}

template <typename T>
inline void atomic_add(T* target, T value) {
#ifdef _OPENMP
  if constexpr (sizeof(T) >= 4) {
#pragma omp atomic
    *target += value;
    return;
  }
#endif
  using I = typename std::conditional<sizeof(T) == 8, std::uint64_t, std::uint32_t>::type;
  if constexpr (sizeof(T) == 2) {
    using H = std::uint16_t;
    H* ip = reinterpret_cast<H*>(target);
    H old_i = __atomic_load_n(ip, __ATOMIC_RELAXED);
    for (;;) {
      T old_d;
      std::memcpy(&old_d, &old_i, sizeof(T));
      T new_d = static_cast<T>(static_cast<float>(old_d) + static_cast<float>(value));
      H new_i;
      std::memcpy(&new_i, &new_d, sizeof(T));
      if (__atomic_compare_exchange_n(ip, &old_i, new_i, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
        return;
      }
    }
  } else {
    I* ip = reinterpret_cast<I*>(target);
    I old_i = __atomic_load_n(ip, __ATOMIC_RELAXED);
    for (;;) {
      T old_d;
      std::memcpy(&old_d, &old_i, sizeof(T));
      T new_d = old_d + value;
      I new_i;
      std::memcpy(&new_i, &new_d, sizeof(T));
      if (__atomic_compare_exchange_n(ip, &old_i, new_i, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
        return;
      }
    }
  }
}

template <typename A>
inline A exp_lookup(const A* table, int ntab, A x) {
  A xs = x * A(-100);
  if (xs < A(0)) {
    return std::exp(x);
  }
  int i = static_cast<int>(static_cast<double>(xs) + 0.5);
  if (i >= ntab) {
    return std::exp(x);
  }
  return table[i];
}

template <typename Fn>
inline void parallel_for(int n, Fn fn) {
  if (n <= 0) {
    return;
  }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    fn(i);
  }
#else
  unsigned nt = std::thread::hardware_concurrency();
  if (nt < 1u) {
    nt = 1u;
  }
  if (nt > static_cast<unsigned>(n)) {
    nt = static_cast<unsigned>(n);
  }
  if (nt <= 1u) {
    for (int i = 0; i < n; ++i) {
      fn(i);
    }
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(nt);
  for (unsigned t = 0; t < nt; ++t) {
    int start = static_cast<int>((static_cast<long long>(t) * n) / nt);
    int end = static_cast<int>((static_cast<long long>(t + 1) * n) / nt);
    pool.emplace_back([=]() {
      for (int i = start; i < end; ++i) {
        fn(i);
      }
    });
  }
  for (auto& th : pool) {
    th.join();
  }
#endif
}

template <typename A>
inline bool inv_det(
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

template <typename A>
inline A iso_gauss(A r2, A s2) {
  return std::pow(A(kTwoPi) * s2, A(-1.5)) * std::exp(A(-0.5) * r2 / s2);
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
inline void stamp_forward(T* grid_re, T* grid_im, const StampIn<T>& in) {
  using A = typename Acc<T>::type;
  const int n0 = in.n0, n1 = in.n1, n2 = in.n2;
  const A nf0 = static_cast<A>(n0);
  const A nf1 = static_cast<A>(n1);
  const A nf2 = static_cast<A>(n2);
  const A o00 = A(in.o[0]), o01 = A(in.o[1]), o02 = A(in.o[2]);
  const A o10 = A(in.o[3]), o11 = A(in.o[4]), o12 = A(in.o[5]);
  const A o20 = A(in.o[6]), o21 = A(in.o[7]), o22 = A(in.o[8]);
  const int k_max = in.k_max;
  const A u_extra = A(in.u_extra);
  const A fs0 = A(in.fs0), fs1 = A(in.fs1), fs2 = A(in.fs2);
  parallel_for(in.n_exp, [&](int i) {
    const A x0 = A(in.xf[i * 3 + 0]);
    const A x1 = A(in.xf[i * 3 + 1]);
    const A x2 = A(in.xf[i * 3 + 2]);
    const A rc = A(in.r_cut[i]);
    const A rcut2 = rc * rc;
    const int h0 = static_cast<int>(std::ceil(static_cast<double>(rc * fs0)));
    const int h1 = static_cast<int>(std::ceil(static_cast<double>(rc * fs1)));
    const int h2 = static_cast<int>(std::ceil(static_cast<double>(rc * fs2)));
    const int g0 = iround(x0 * nf0);
    const int g1 = iround(x1 * nf1);
    const int g2 = iround(x2 * nf2);
    const A wi = A(in.w[i]);
    const A cfp = A(in.c[i]) + A(in.fp[i]);
    const A fd = A(in.fdp[i]);
    if (!in.aniso[i]) {
      const A s2b = A(in.u_iso[i]) + u_extra;
      const int nterm = k_max + 1;
      A as_real[kMaxTerms];
      A bs_real[kMaxTerms];
      for (int kk = 0; kk < k_max; ++kk) {
        const A s2 = s2b + A(in.b[i * k_max + kk]) / A(kEightPi2);
        as_real[kk] = wi * A(in.a[i * k_max + kk]) * std::pow(A(kTwoPi) * s2, A(-1.5));
        bs_real[kk] = A(-0.5) / s2;
      }
      as_real[k_max] = wi * cfp * std::pow(A(kTwoPi) * s2b, A(-1.5));
      bs_real[k_max] = A(-0.5) / s2b;
      const A as_im = in.has_imag ? wi * fd * std::pow(A(kTwoPi) * s2b, A(-1.5)) : A(0);
      for (int di = -h0; di <= h0; ++di) {
        const int gi0 = wrap(g0 + di, n0);
        const A rf0 = (g0 + di) / nf0 - x0;
        const A t0x = o00 * rf0, t0y = o10 * rf0, t0z = o20 * rf0;
        const int row0 = gi0 * n1;
        for (int dj = -h1; dj <= h1; ++dj) {
          const int gi1 = wrap(g1 + dj, n1);
          const A rf1 = (g1 + dj) / nf1 - x1;
          const A t1x = o01 * rf1, t1y = o11 * rf1, t1z = o21 * rf1;
          const int row01 = (row0 + gi1) * n2;
          for (int dk = -h2; dk <= h2; ++dk) {
            const int gi2 = wrap(g2 + dk, n2);
            const A rf2 = (g2 + dk) / nf2 - x2;
            const A rcx = t0x + t1x + o02 * rf2;
            const A rcy = t0y + t1y + o12 * rf2;
            const A rcz = t0z + t1z + o22 * rf2;
            const A r2 = rcx * rcx + rcy * rcy + rcz * rcz;
            if (r2 > rcut2) {
              continue;
            }
            A contr = A(0);
            for (int kk = 0; kk < nterm; ++kk) {
              contr += as_real[kk] * exp_lookup(in.exp_table, in.ntab, bs_real[kk] * r2);
            }
            const int flat = row01 + gi2;
            atomic_add(grid_re + flat, T(contr));
            if (in.has_imag) {
              atomic_add(grid_im + flat, T(as_im * exp_lookup(in.exp_table, in.ntab, bs_real[k_max] * r2)));
            }
          }
        }
      }
    } else {
      const T* uc = in.u_cart + i * 9;
      const A u00 = A(uc[0]), u01 = A(uc[1]), u02 = A(uc[2]);
      const A u11 = A(uc[4]), u12 = A(uc[5]), u22 = A(uc[8]);
      for (int di = -h0; di <= h0; ++di) {
        const int gi0 = wrap(g0 + di, n0);
        const A rf0 = (g0 + di) / nf0 - x0;
        const A t0x = o00 * rf0, t0y = o10 * rf0, t0z = o20 * rf0;
        const int row0 = gi0 * n1;
        for (int dj = -h1; dj <= h1; ++dj) {
          const int gi1 = wrap(g1 + dj, n1);
          const A rf1 = (g1 + dj) / nf1 - x1;
          const A t1x = o01 * rf1, t1y = o11 * rf1, t1z = o21 * rf1;
          const int row01 = (row0 + gi1) * n2;
          for (int dk = -h2; dk <= h2; ++dk) {
            const int gi2 = wrap(g2 + dk, n2);
            const A rf2 = (g2 + dk) / nf2 - x2;
            const A rcx = t0x + t1x + o02 * rf2;
            const A rcy = t0y + t1y + o12 * rf2;
            const A rcz = t0z + t1z + o22 * rf2;
            const A r2 = rcx * rcx + rcy * rcy + rcz * rcz;
            if (r2 > rcut2) {
              continue;
            }
            A p00, p01, p02, p11, p12, p22, det;
            if (!inv_det(u00 + u_extra, u01, u02, u11 + u_extra, u12, u22 + u_extra, p00, p01, p02, p11, p12, p22, det)) {
              continue;
            }
            const A q = p00 * rcx * rcx + p11 * rcy * rcy + p22 * rcz * rcz +
                        A(2) * (p01 * rcx * rcy + p02 * rcx * rcz + p12 * rcy * rcz);
            const A base = A(kNorm0) * std::pow(det, A(-0.5)) * exp_lookup(in.exp_table, in.ntab, A(-0.5) * q);
            A terms = A(0);
            for (int kk = 0; kk < k_max; ++kk) {
              const A extra = A(in.b[i * k_max + kk]) / A(kEightPi2) + u_extra;
              if (!inv_det(u00 + extra, u01, u02, u11 + extra, u12, u22 + extra, p00, p01, p02, p11, p12, p22, det)) {
                continue;
              }
              const A qk = p00 * rcx * rcx + p11 * rcy * rcy + p22 * rcz * rcz +
                           A(2) * (p01 * rcx * rcy + p02 * rcx * rcz + p12 * rcy * rcz);
              terms += A(in.a[i * k_max + kk]) * (A(kNorm0) * std::pow(det, A(-0.5)) * exp_lookup(in.exp_table, in.ntab, A(-0.5) * qk));
            }
            const int flat = row01 + gi2;
            atomic_add(grid_re + flat, T((terms + cfp * base) * wi));
            if (in.has_imag) {
              atomic_add(grid_im + flat, T(fd * base * wi));
            }
          }
        }
      }
    }
  });
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
inline void stamp_vjp(const T* gre, const T* gim, const StampIn<T>& in, const VjpOut<T>& out) {
  using A = typename Acc<T>::type;
  const int n0 = in.n0, n1 = in.n1, n2 = in.n2;
  const A nf0 = static_cast<A>(n0);
  const A nf1 = static_cast<A>(n1);
  const A nf2 = static_cast<A>(n2);
  const A o00 = A(in.o[0]), o01 = A(in.o[1]), o02 = A(in.o[2]);
  const A o10 = A(in.o[3]), o11 = A(in.o[4]), o12 = A(in.o[5]);
  const A o20 = A(in.o[6]), o21 = A(in.o[7]), o22 = A(in.o[8]);
  const int k_max = in.k_max;
  const A u_extra = A(in.u_extra);
  const A fs0 = A(in.fs0), fs1 = A(in.fs1), fs2 = A(in.fs2);
  parallel_for(in.n_exp, [&](int i) {
    const A x0 = A(in.xf[i * 3 + 0]);
    const A x1 = A(in.xf[i * 3 + 1]);
    const A x2 = A(in.xf[i * 3 + 2]);
    const A rc = A(in.r_cut[i]);
    const A rcut2 = rc * rc;
    const int h0 = static_cast<int>(std::ceil(static_cast<double>(rc * fs0)));
    const int h1 = static_cast<int>(std::ceil(static_cast<double>(rc * fs1)));
    const int h2 = static_cast<int>(std::ceil(static_cast<double>(rc * fs2)));
    const int g0 = iround(x0 * nf0);
    const int g1 = iround(x1 * nf1);
    const int g2 = iround(x2 * nf2);
    const A wi = A(in.w[i]);
    const A cfp = A(in.c[i]) + A(in.fp[i]);
    const A fd = A(in.fdp[i]);
    A acc_x0 = A(0), acc_x1 = A(0), acc_x2 = A(0), acc_u = A(0);
    A acc_uc00 = A(0), acc_uc01 = A(0), acc_uc02 = A(0), acc_uc11 = A(0), acc_uc12 = A(0), acc_uc22 = A(0);
    A acc_w = A(0), acc_fp = A(0), acc_fdp = A(0);
    if (!in.aniso[i]) {
      const A s2b = A(in.u_iso[i]) + u_extra;
      for (int di = -h0; di <= h0; ++di) {
        const int gi0 = g0 + di;
        const A rf0 = gi0 / nf0 - x0;
        for (int dj = -h1; dj <= h1; ++dj) {
          const int gi1 = g1 + dj;
          const A rf1 = gi1 / nf1 - x1;
          for (int dk = -h2; dk <= h2; ++dk) {
            const int gi2 = g2 + dk;
            const A rf2 = gi2 / nf2 - x2;
            const A rcx = o00 * rf0 + o01 * rf1 + o02 * rf2;
            const A rcy = o10 * rf0 + o11 * rf1 + o12 * rf2;
            const A rcz = o20 * rf0 + o21 * rf1 + o22 * rf2;
            const A r2 = rcx * rcx + rcy * rcy + rcz * rcz;
            if (r2 > rcut2) {
              continue;
            }
            const A base = iso_gauss(r2, s2b);
            A terms = A(0), d_terms_dr2 = A(0), d_terms_ds2 = A(0);
            for (int kk = 0; kk < k_max; ++kk) {
              const A s2k = s2b + A(in.b[i * k_max + kk]) / A(kEightPi2);
              const A gk = iso_gauss(r2, s2k);
              const A ak = A(in.a[i * k_max + kk]);
              terms += ak * gk;
              d_terms_dr2 += ak * gk * (A(-0.5) / s2k);
              d_terms_ds2 += ak * gk * (A(0.5) * r2 / (s2k * s2k) - A(1.5) / s2k);
            }
            const A d_base_dr2 = base * (A(-0.5) / s2b);
            const A d_base_ds2 = base * (A(0.5) * r2 / (s2b * s2b) - A(1.5) / s2b);
            const int flat = (wrap(gi0, n0) * n1 + wrap(gi1, n1)) * n2 + wrap(gi2, n2);
            const A er = A(gre[flat]);
            const A ei = in.has_imag ? A(gim[flat]) : A(0);
            const A rho_re = terms + cfp * base;
            acc_w += er * rho_re + ei * (fd * base);
            acc_fp += er * wi * base;
            acc_fdp += ei * wi * base;
            const A dL_dr2 = er * wi * (d_terms_dr2 + cfp * d_base_dr2) + ei * wi * fd * d_base_dr2;
            const A ot0 = o00 * rcx + o10 * rcy + o20 * rcz;
            const A ot1 = o01 * rcx + o11 * rcy + o21 * rcz;
            const A ot2 = o02 * rcx + o12 * rcy + o22 * rcz;
            acc_x0 += dL_dr2 * (A(-2) * ot0);
            acc_x1 += dL_dr2 * (A(-2) * ot1);
            acc_x2 += dL_dr2 * (A(-2) * ot2);
            acc_u += er * wi * (d_terms_ds2 + cfp * d_base_ds2) + ei * wi * fd * d_base_ds2;
          }
        }
      }
    } else {
      const T* uc = in.u_cart + i * 9;
      const A u00 = A(uc[0]), u01 = A(uc[1]), u02 = A(uc[2]);
      const A u11 = A(uc[4]), u12 = A(uc[5]), u22 = A(uc[8]);
      for (int di = -h0; di <= h0; ++di) {
        const int gi0 = g0 + di;
        const A rf0 = gi0 / nf0 - x0;
        for (int dj = -h1; dj <= h1; ++dj) {
          const int gi1 = g1 + dj;
          const A rf1 = gi1 / nf1 - x1;
          for (int dk = -h2; dk <= h2; ++dk) {
            const int gi2 = g2 + dk;
            const A rf2 = gi2 / nf2 - x2;
            const A rcx = o00 * rf0 + o01 * rf1 + o02 * rf2;
            const A rcy = o10 * rf0 + o11 * rf1 + o12 * rf2;
            const A rcz = o20 * rf0 + o21 * rf1 + o22 * rf2;
            const A r2 = rcx * rcx + rcy * rcy + rcz * rcz;
            if (r2 > rcut2) {
              continue;
            }
            const int flat = (wrap(gi0, n0) * n1 + wrap(gi1, n1)) * n2 + wrap(gi2, n2);
            const A er = A(gre[flat]);
            const A ei = in.has_imag ? A(gim[flat]) : A(0);
            A p00, p01, p02, p11, p12, p22, det;
            if (!inv_det(u00 + u_extra, u01, u02, u11 + u_extra, u12, u22 + u_extra, p00, p01, p02, p11, p12, p22, det)) {
              continue;
            }
            const A q = p00 * rcx * rcx + p11 * rcy * rcy + p22 * rcz * rcz +
                        A(2) * (p01 * rcx * rcy + p02 * rcx * rcz + p12 * rcy * rcz);
            const A base = A(kNorm0) * std::pow(det, A(-0.5)) * std::exp(A(-0.5) * q);
            const A pr0 = p00 * rcx + p01 * rcy + p02 * rcz;
            const A pr1 = p01 * rcx + p11 * rcy + p12 * rcz;
            const A pr2 = p02 * rcx + p12 * rcy + p22 * rcz;
            A terms = A(0), dx0 = A(0), dx1 = A(0), dx2 = A(0);
            A duc00 = A(0), duc01 = A(0), duc02 = A(0), duc11 = A(0), duc12 = A(0), duc22 = A(0);
            const A scale_base = er * wi * cfp + ei * wi * fd;
            dx0 += scale_base * base * (o00 * pr0 + o10 * pr1 + o20 * pr2);
            dx1 += scale_base * base * (o01 * pr0 + o11 * pr1 + o21 * pr2);
            dx2 += scale_base * base * (o02 * pr0 + o12 * pr1 + o22 * pr2);
            const A sb = scale_base * base;
            duc00 += sb * (A(-0.5) * p00 + A(0.5) * pr0 * pr0);
            duc01 += sb * (A(-0.5) * p01 + A(0.5) * pr0 * pr1);
            duc02 += sb * (A(-0.5) * p02 + A(0.5) * pr0 * pr2);
            duc11 += sb * (A(-0.5) * p11 + A(0.5) * pr1 * pr1);
            duc12 += sb * (A(-0.5) * p12 + A(0.5) * pr1 * pr2);
            duc22 += sb * (A(-0.5) * p22 + A(0.5) * pr2 * pr2);
            for (int kk = 0; kk < k_max; ++kk) {
              const A extra = A(in.b[i * k_max + kk]) / A(kEightPi2) + u_extra;
              if (!inv_det(u00 + extra, u01, u02, u11 + extra, u12, u22 + extra, p00, p01, p02, p11, p12, p22, det)) {
                continue;
              }
              const A qk = p00 * rcx * rcx + p11 * rcy * rcy + p22 * rcz * rcz +
                           A(2) * (p01 * rcx * rcy + p02 * rcx * rcz + p12 * rcy * rcz);
              const A gk = A(kNorm0) * std::pow(det, A(-0.5)) * std::exp(A(-0.5) * qk);
              const A prk0 = p00 * rcx + p01 * rcy + p02 * rcz;
              const A prk1 = p01 * rcx + p11 * rcy + p12 * rcz;
              const A prk2 = p02 * rcx + p12 * rcy + p22 * rcz;
              terms += A(in.a[i * k_max + kk]) * gk;
              const A sk = er * wi * A(in.a[i * k_max + kk]);
              dx0 += sk * gk * (o00 * prk0 + o10 * prk1 + o20 * prk2);
              dx1 += sk * gk * (o01 * prk0 + o11 * prk1 + o21 * prk2);
              dx2 += sk * gk * (o02 * prk0 + o12 * prk1 + o22 * prk2);
              const A sg = sk * gk;
              duc00 += sg * (A(-0.5) * p00 + A(0.5) * prk0 * prk0);
              duc01 += sg * (A(-0.5) * p01 + A(0.5) * prk0 * prk1);
              duc02 += sg * (A(-0.5) * p02 + A(0.5) * prk0 * prk2);
              duc11 += sg * (A(-0.5) * p11 + A(0.5) * prk1 * prk1);
              duc12 += sg * (A(-0.5) * p12 + A(0.5) * prk1 * prk2);
              duc22 += sg * (A(-0.5) * p22 + A(0.5) * prk2 * prk2);
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
  });
}

}  // namespace phridge_stamp
