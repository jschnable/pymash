#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

double kLog2Pi = std::log(2.0 * M_PI);

double clamp_min(double x, double lo) {
    return (x < lo) ? lo : x;
}

bool cholesky_decompose(const std::vector<double>& A, std::vector<double>& L, std::size_t d) {
    L.assign(d * d, 0.0);
    for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = A[i * d + j];
            for (std::size_t k = 0; k < j; ++k) {
                sum -= L[i * d + k] * L[j * d + k];
            }
            if (i == j) {
                if (!(sum > 0.0) || !std::isfinite(sum)) {
                    return false;
                }
                L[i * d + j] = std::sqrt(sum);
            } else {
                L[i * d + j] = sum / L[j * d + j];
            }
        }
    }
    return true;
}

bool cholesky_with_jitter(std::vector<double> A, std::vector<double>& L, std::size_t d) {
    if (cholesky_decompose(A, L, d)) {
        return true;
    }
    double jitter = 1e-10;
    for (int t = 0; t < 10; ++t) {
        for (std::size_t i = 0; i < d; ++i) {
            A[i * d + i] += jitter;
        }
        if (cholesky_decompose(A, L, d)) {
            return true;
        }
        jitter *= 10.0;
    }
    return false;
}

std::vector<double> solve_lower(const std::vector<double>& L, const std::vector<double>& b, std::size_t d) {
    std::vector<double> x(d, 0.0);
    for (std::size_t i = 0; i < d; ++i) {
        double sum = b[i];
        for (std::size_t k = 0; k < i; ++k) {
            sum -= L[i * d + k] * x[k];
        }
        x[i] = sum / L[i * d + i];
    }
    return x;
}

std::vector<double> solve_upper_from_lower_transpose(const std::vector<double>& L, const std::vector<double>& b, std::size_t d) {
    std::vector<double> x(d, 0.0);
    for (std::size_t ii = 0; ii < d; ++ii) {
        std::size_t i = d - 1 - ii;
        double sum = b[i];
        for (std::size_t k = i + 1; k < d; ++k) {
            sum -= L[k * d + i] * x[k];
        }
        x[i] = sum / L[i * d + i];
    }
    return x;
}

std::vector<double> solve_spd_vector(const std::vector<double>& L, const std::vector<double>& b, std::size_t d) {
    auto y = solve_lower(L, b, d);
    return solve_upper_from_lower_transpose(L, y, d);
}

void solve_spd_vector_inplace(
    const std::vector<double>& L,
    const std::vector<double>& b,
    std::size_t d,
    std::vector<double>& y,
    std::vector<double>& x
) {
    y.resize(d);
    x.resize(d);
    for (std::size_t i = 0; i < d; ++i) {
        double sum = b[i];
        for (std::size_t k = 0; k < i; ++k) {
            sum -= L[i * d + k] * y[k];
        }
        y[i] = sum / L[i * d + i];
    }
    for (std::size_t ii = 0; ii < d; ++ii) {
        std::size_t i = d - 1 - ii;
        double sum = y[i];
        for (std::size_t k = i + 1; k < d; ++k) {
            sum -= L[k * d + i] * x[k];
        }
        x[i] = sum / L[i * d + i];
    }
}

void solve_spd_matrix_cols_inplace(
    const std::vector<double>& L,
    const std::vector<double>& B,
    std::size_t d,
    std::size_t m,
    std::vector<double>& X,
    std::vector<double>& col,
    std::vector<double>& y_tmp,
    std::vector<double>& x_tmp
) {
    X.assign(d * m, 0.0);
    col.resize(d);
    for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t i = 0; i < d; ++i) {
            col[i] = B[i * m + j];
        }
        solve_spd_vector_inplace(L, col, d, y_tmp, x_tmp);
        for (std::size_t i = 0; i < d; ++i) {
            X[i * m + j] = x_tmp[i];
        }
    }
}

void solve_spd_matrix_inplace(
    const std::vector<double>& L,
    const std::vector<double>& B,
    std::size_t d,
    std::vector<double>& X,
    std::vector<double>& col,
    std::vector<double>& y_tmp,
    std::vector<double>& x_tmp
) {
    X.assign(d * d, 0.0);
    col.resize(d);
    for (std::size_t j = 0; j < d; ++j) {
        for (std::size_t i = 0; i < d; ++i) {
            col[i] = B[i * d + j];
        }
        solve_spd_vector_inplace(L, col, d, y_tmp, x_tmp);
        for (std::size_t i = 0; i < d; ++i) {
            X[i * d + j] = x_tmp[i];
        }
    }
}

std::vector<double> solve_spd_matrix(const std::vector<double>& L, const std::vector<double>& B, std::size_t d) {
    std::vector<double> X(d * d, 0.0);
    std::vector<double> col(d);
    for (std::size_t j = 0; j < d; ++j) {
        for (std::size_t i = 0; i < d; ++i) {
            col[i] = B[i * d + j];
        }
        auto x = solve_spd_vector(L, col, d);
        for (std::size_t i = 0; i < d; ++i) {
            X[i * d + j] = x[i];
        }
    }
    return X;
}

std::vector<double> matmul_square(const std::vector<double>& A, const std::vector<double>& B, std::size_t d) {
    std::vector<double> C(d * d, 0.0);
    for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t k = 0; k < d; ++k) {
            double aik = A[i * d + k];
            for (std::size_t j = 0; j < d; ++j) {
                C[i * d + j] += aik * B[k * d + j];
            }
        }
    }
    return C;
}

std::vector<double> matvec_square(const std::vector<double>& A, const std::vector<double>& x, std::size_t d) {
    std::vector<double> y(d, 0.0);
    for (std::size_t i = 0; i < d; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < d; ++j) {
            sum += A[i * d + j] * x[j];
        }
        y[i] = sum;
    }
    return y;
}

std::vector<double> solve_spd_matrix_cols(
    const std::vector<double>& L,
    const std::vector<double>& B,
    std::size_t d,
    std::size_t m
) {
    std::vector<double> X(d * m, 0.0);
    std::vector<double> col(d);
    for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t i = 0; i < d; ++i) {
            col[i] = B[i * m + j];
        }
        auto x = solve_spd_vector(L, col, d);
        for (std::size_t i = 0; i < d; ++i) {
            X[i * m + j] = x[i];
        }
    }
    return X;
}

std::vector<double> matmul_rect(
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::size_t n,
    std::size_t k,
    std::size_t m
) {
    std::vector<double> C(n * m, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t kk = 0; kk < k; ++kk) {
            double aik = A[i * k + kk];
            for (std::size_t j = 0; j < m; ++j) {
                C[i * m + j] += aik * B[kk * m + j];
            }
        }
    }
    return C;
}

std::vector<double> matmul_transpose_left_rect(
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::size_t n,
    std::size_t m,
    std::size_t k
) {
    // A: (n x m), B: (n x k), returns A^T B: (m x k)
    std::vector<double> C(m * k, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t row = 0; row < n; ++row) {
            double ari = A[row * m + i];
            for (std::size_t j = 0; j < k; ++j) {
                C[i * k + j] += ari * B[row * k + j];
            }
        }
    }
    return C;
}

std::vector<double> matvec_transpose_rect(
    const std::vector<double>& A,
    const std::vector<double>& x,
    std::size_t n,
    std::size_t m
) {
    // A: (n x m), x: (n), returns A^T x: (m)
    std::vector<double> y(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        double sum = 0.0;
        for (std::size_t row = 0; row < n; ++row) {
            sum += A[row * m + i] * x[row];
        }
        y[i] = sum;
    }
    return y;
}

double log_mvn_from_chol_with_obs(
    const std::vector<double>& L,
    const std::vector<double>& y,
    std::size_t d
) {
    std::vector<double> z(d, 0.0);
    double quad = 0.0;
    double logdet = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
        double sum = y[i];
        for (std::size_t k = 0; k < i; ++k) {
            sum -= L[i * d + k] * z[k];
        }
        z[i] = sum / L[i * d + i];
        quad += z[i] * z[i];
        logdet += std::log(L[i * d + i]);
    }
    logdet *= 2.0;
    return -0.5 * (static_cast<double>(d) * kLog2Pi + logdet + quad);
}

bool solve_linear_system(
    const std::vector<double>& A_in,
    std::size_t n,
    const std::vector<double>& B_in,
    std::size_t m,
    std::vector<double>& X_out
) {
    const double eps = 1e-14;
    std::vector<double> A = A_in;
    std::vector<double> B = B_in;

    for (std::size_t k = 0; k < n; ++k) {
        std::size_t piv = k;
        double piv_abs = std::abs(A[k * n + k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            double v = std::abs(A[i * n + k]);
            if (v > piv_abs) {
                piv_abs = v;
                piv = i;
            }
        }
        if (!(piv_abs > eps)) {
            return false;
        }
        if (piv != k) {
            for (std::size_t c = 0; c < n; ++c) {
                std::swap(A[k * n + c], A[piv * n + c]);
            }
            for (std::size_t c = 0; c < m; ++c) {
                std::swap(B[k * m + c], B[piv * m + c]);
            }
        }

        double pivot = A[k * n + k];
        for (std::size_t c = k; c < n; ++c) {
            A[k * n + c] /= pivot;
        }
        for (std::size_t c = 0; c < m; ++c) {
            B[k * m + c] /= pivot;
        }

        for (std::size_t i = 0; i < n; ++i) {
            if (i == k) {
                continue;
            }
            double f = A[i * n + k];
            if (f == 0.0) {
                continue;
            }
            for (std::size_t c = k; c < n; ++c) {
                A[i * n + c] -= f * A[k * n + c];
            }
            for (std::size_t c = 0; c < m; ++c) {
                B[i * m + c] -= f * B[k * m + c];
            }
        }
    }

    X_out = std::move(B);
    return true;
}

bool invert_with_jitter(const std::vector<double>& A, std::size_t n, std::vector<double>& Ainv) {
    std::vector<double> I(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        I[i * n + i] = 1.0;
    }

    if (solve_linear_system(A, n, I, n, Ainv)) {
        return true;
    }

    std::vector<double> Aj = A;
    double jitter = 1e-12;
    for (int t = 0; t < 8; ++t) {
        for (std::size_t i = 0; i < n; ++i) {
            Aj[i * n + i] += jitter;
        }
        if (solve_linear_system(Aj, n, I, n, Ainv)) {
            return true;
        }
        jitter *= 10.0;
    }
    return false;
}

double ndtr_scalar(double z) {
    constexpr double inv_sqrt2 = 0.70710678118654752440;
    return 0.5 * std::erfc(-z * inv_sqrt2);
}

double log_mvn_from_chol(const std::vector<double>& L, const std::vector<double>& y, std::size_t d) {
    auto z = solve_lower(L, y, d);
    double quad = 0.0;
    double logdet = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
        quad += z[i] * z[i];
        logdet += std::log(L[i * d + i]);
    }
    logdet *= 2.0;
    return -0.5 * (static_cast<double>(d) * kLog2Pi + logdet + quad);
}

void symmetrize_in_place(std::vector<double>& A, std::size_t d) {
    for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t j = i + 1; j < d; ++j) {
            double v = 0.5 * (A[i * d + j] + A[j * d + i]);
            A[i * d + j] = v;
            A[j * d + i] = v;
        }
    }
}

void ensure_choleskyable(std::vector<double>& A, std::size_t d) {
    std::vector<double> L;
    if (cholesky_with_jitter(A, L, d)) {
        return;
    }
    for (std::size_t i = 0; i < d; ++i) {
        A[i * d + i] += 1e-3;
    }
}

double log_mvn_cov_plus_u_woodbury(
    const std::vector<double>& chol_v,
    double logdet_v,
    double quad_v,
    const std::vector<double>& vinv_y,
    const std::vector<double>& u_factor,
    std::size_t d,
    std::size_t r,
    bool& ok
) {
    // u_factor is (d x r), where U = u_factor * u_factor^T
    auto vinv_u = solve_spd_matrix_cols(chol_v, u_factor, d, r);      // (d x r)
    auto m = matmul_transpose_left_rect(u_factor, vinv_u, d, r, r);    // (r x r)
    for (std::size_t i = 0; i < r; ++i) {
        m[i * r + i] += 1.0;
    }

    std::vector<double> chol_m;
    if (!cholesky_with_jitter(m, chol_m, r)) {
        ok = false;
        return -std::numeric_limits<double>::infinity();
    }

    auto b = matvec_transpose_rect(u_factor, vinv_y, d, r);
    auto minv_b = solve_spd_vector(chol_m, b, r);
    double correction = 0.0;
    for (std::size_t i = 0; i < r; ++i) {
        correction += b[i] * minv_b[i];
    }

    double logdet_m = 0.0;
    for (std::size_t i = 0; i < r; ++i) {
        logdet_m += std::log(chol_m[i * r + i]);
    }
    logdet_m *= 2.0;

    double quad = quad_v - correction;
    double logdet = logdet_v + logdet_m;
    ok = std::isfinite(quad) && std::isfinite(logdet);
    if (!ok) {
        return -std::numeric_limits<double>::infinity();
    }
    return -0.5 * (static_cast<double>(d) * kLog2Pi + logdet + quad);
}

} // namespace

py::dict fit_extreme_deconvolution(
    py::array_t<double, py::array::c_style | py::array::forcecast> ydata,
    py::array_t<double, py::array::c_style | py::array::forcecast> ycovar,
    py::array_t<double, py::array::c_style | py::array::forcecast> xamp,
    py::array_t<double, py::array::c_style | py::array::forcecast> xcovar,
    int maxiter = 500,
    double tol = 1e-6,
    double w = 0.0,
    bool verbose = false
) {
    auto ybuf = ydata.request();
    auto sbuf = ycovar.request();
    auto abuf = xamp.request();
    auto cbuf = xcovar.request();

    if (ybuf.ndim != 2) {
        throw std::invalid_argument("ydata must be 2D (N, D)");
    }
    if (abuf.ndim != 1) {
        throw std::invalid_argument("xamp must be 1D (K)");
    }
    if (cbuf.ndim != 3) {
        throw std::invalid_argument("xcovar must be 3D (K, D, D)");
    }

    std::size_t N = static_cast<std::size_t>(ybuf.shape[0]);
    std::size_t D = static_cast<std::size_t>(ybuf.shape[1]);
    std::size_t K = static_cast<std::size_t>(abuf.shape[0]);

    if (cbuf.shape[0] != static_cast<py::ssize_t>(K) ||
        cbuf.shape[1] != static_cast<py::ssize_t>(D) ||
        cbuf.shape[2] != static_cast<py::ssize_t>(D)) {
        throw std::invalid_argument("xcovar must have shape (K, D, D)");
    }

    std::vector<double> Y(static_cast<std::size_t>(ybuf.size));
    std::copy_n(static_cast<double*>(ybuf.ptr), ybuf.size, Y.begin());

    std::vector<double> S(N * D * D, 0.0);
    if (sbuf.ndim == 2) {
        if (sbuf.shape[0] != static_cast<py::ssize_t>(N) ||
            sbuf.shape[1] != static_cast<py::ssize_t>(D)) {
            throw std::invalid_argument("2D ycovar must have shape (N, D)");
        }
        auto* s_ptr = static_cast<double*>(sbuf.ptr);
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t r = 0; r < D; ++r) {
                double v = s_ptr[i * D + r];
                S[i * D * D + r * D + r] = v;
            }
        }
    } else if (sbuf.ndim == 3) {
        auto* s_ptr = static_cast<double*>(sbuf.ptr);
        bool shape_ndd =
            sbuf.shape[0] == static_cast<py::ssize_t>(N) &&
            sbuf.shape[1] == static_cast<py::ssize_t>(D) &&
            sbuf.shape[2] == static_cast<py::ssize_t>(D);
        bool shape_ddn =
            sbuf.shape[0] == static_cast<py::ssize_t>(D) &&
            sbuf.shape[1] == static_cast<py::ssize_t>(D) &&
            sbuf.shape[2] == static_cast<py::ssize_t>(N);

        if (!shape_ndd && !shape_ddn) {
            throw std::invalid_argument("3D ycovar must have shape (N, D, D) or (D, D, N)");
        }

        if (shape_ndd) {
            std::copy_n(s_ptr, sbuf.size, S.begin());
        } else {
            for (std::size_t i = 0; i < N; ++i) {
                for (std::size_t r = 0; r < D; ++r) {
                    for (std::size_t c = 0; c < D; ++c) {
                        S[i * D * D + r * D + c] = s_ptr[r * D * N + c * N + i];
                    }
                }
            }
        }
    } else {
        throw std::invalid_argument("ycovar must be 2D or 3D");
    }

    std::vector<double> amp(K);
    std::copy_n(static_cast<double*>(abuf.ptr), abuf.size, amp.begin());
    for (double& a : amp) {
        a = clamp_min(a, std::numeric_limits<double>::min());
    }
    double amp_sum = 0.0;
    for (double a : amp) {
        amp_sum += a;
    }
    for (double& a : amp) {
        a /= amp_sum;
    }

    std::vector<std::vector<double>> cov(K, std::vector<double>(D * D, 0.0));
    auto* c_ptr = static_cast<double*>(cbuf.ptr);
    for (std::size_t k = 0; k < K; ++k) {
        std::copy_n(c_ptr + k * D * D, D * D, cov[k].begin());
        symmetrize_in_place(cov[k], D);
        ensure_choleskyable(cov[k], D);
    }

    std::vector<double> responsibilities(N * K, 0.0);
    std::vector<double> objective;
    objective.reserve(static_cast<std::size_t>(maxiter));

    for (int iter = 0; iter < maxiter; ++iter) {
        std::vector<double> amp_old = amp;
        double loglik_sum = 0.0;

        for (std::size_t i = 0; i < N; ++i) {
            const double* yi = &Y[i * D];
            std::vector<double> yvec(yi, yi + D);
            const double* si = &S[i * D * D];

            std::vector<double> logp(K, -std::numeric_limits<double>::infinity());
            for (std::size_t k = 0; k < K; ++k) {
                std::vector<double> T(D * D, 0.0);
                for (std::size_t idx = 0; idx < D * D; ++idx) {
                    T[idx] = cov[k][idx] + si[idx];
                }
                std::vector<double> L;
                if (!cholesky_with_jitter(T, L, D)) {
                    continue;
                }
                double ll = log_mvn_from_chol(L, yvec, D);
                logp[k] = std::log(clamp_min(amp[k], std::numeric_limits<double>::min())) + ll;
            }

            double maxlog = *std::max_element(logp.begin(), logp.end());
            double denom = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                double e = std::exp(logp[k] - maxlog);
                responsibilities[i * K + k] = e;
                denom += e;
            }
            denom = clamp_min(denom, std::numeric_limits<double>::min());
            for (std::size_t k = 0; k < K; ++k) {
                responsibilities[i * K + k] /= denom;
            }
            loglik_sum += maxlog + std::log(denom);
        }

        std::vector<double> Nk(K, 0.0);
        std::vector<std::vector<double>> cov_acc(K, std::vector<double>(D * D, 0.0));

        for (std::size_t i = 0; i < N; ++i) {
            const double* yi = &Y[i * D];
            std::vector<double> yvec(yi, yi + D);
            const double* si = &S[i * D * D];

            for (std::size_t k = 0; k < K; ++k) {
                double rik = responsibilities[i * K + k];
                if (rik < 1e-15) {
                    continue;
                }

                std::vector<double> T(D * D, 0.0);
                for (std::size_t idx = 0; idx < D * D; ++idx) {
                    T[idx] = cov[k][idx] + si[idx];
                }
                std::vector<double> L;
                if (!cholesky_with_jitter(T, L, D)) {
                    continue;
                }

                auto t_inv_y = solve_spd_vector(L, yvec, D);
                auto m = matvec_square(cov[k], t_inv_y, D);

                auto t_inv_v = solve_spd_matrix(L, cov[k], D);
                auto v_t_inv_v = matmul_square(cov[k], t_inv_v, D);

                Nk[k] += rik;
                for (std::size_t r = 0; r < D; ++r) {
                    for (std::size_t c = 0; c < D; ++c) {
                        double c_rc = cov[k][r * D + c] - v_t_inv_v[r * D + c];
                        double mm = m[r] * m[c];
                        cov_acc[k][r * D + c] += rik * (c_rc + mm);
                    }
                }
            }
        }

        for (std::size_t k = 0; k < K; ++k) {
            if (Nk[k] <= 1e-12) {
                amp[k] = 1e-12;
                continue;
            }
            amp[k] = Nk[k] / static_cast<double>(N);

            for (std::size_t idx = 0; idx < D * D; ++idx) {
                cov[k][idx] = cov_acc[k][idx] / Nk[k];
            }
            symmetrize_in_place(cov[k], D);
            if (w > 0.0) {
                for (std::size_t r = 0; r < D; ++r) {
                    cov[k][r * D + r] += w;
                }
            }
            ensure_choleskyable(cov[k], D);
        }

        double new_amp_sum = 0.0;
        for (double a : amp) {
            new_amp_sum += a;
        }
        for (double& a : amp) {
            a = clamp_min(a / new_amp_sum, std::numeric_limits<double>::min());
        }
        double renorm = 0.0;
        for (double a : amp) {
            renorm += a;
        }
        for (double& a : amp) {
            a /= renorm;
        }

        double avg_loglik = loglik_sum / static_cast<double>(N);
        objective.push_back(avg_loglik);

        double maxd = 0.0;
        for (std::size_t k = 0; k < K; ++k) {
            maxd = std::max(maxd, std::abs(amp[k] - amp_old[k]));
        }

        if (verbose && (iter % 50 == 0 || maxd < tol)) {
            py::print("[extreme_deconvolution_cpp] iter=", iter, " avgloglik=", avg_loglik, " maxd=", maxd);
        }
        if (maxd < tol) {
            break;
        }
    }

    py::array_t<double> out_amp(static_cast<py::ssize_t>(K));
    auto out_amp_m = out_amp.mutable_unchecked<1>();
    for (std::size_t k = 0; k < K; ++k) {
        out_amp_m(static_cast<py::ssize_t>(k)) = amp[k];
    }

    py::array_t<double> out_cov({static_cast<py::ssize_t>(K), static_cast<py::ssize_t>(D), static_cast<py::ssize_t>(D)});
    auto out_cov_m = out_cov.mutable_unchecked<3>();
    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t r = 0; r < D; ++r) {
            for (std::size_t c = 0; c < D; ++c) {
                out_cov_m(static_cast<py::ssize_t>(k), static_cast<py::ssize_t>(r), static_cast<py::ssize_t>(c)) =
                    cov[k][r * D + c];
            }
        }
    }

    py::array_t<double> out_objective(static_cast<py::ssize_t>(objective.size()));
    auto out_obj_m = out_objective.mutable_unchecked<1>();
    for (std::size_t i = 0; i < objective.size(); ++i) {
        out_obj_m(static_cast<py::ssize_t>(i)) = objective[i];
    }

    py::dict out;
    out["xamp"] = out_amp;
    out["xcovar"] = out_cov;
    out["avgloglikedata"] = objective.empty() ? -std::numeric_limits<double>::infinity() : objective.back();
    out["objective"] = out_objective;
    out["status"] = 0;
    return out;
}

py::array_t<double> calc_lik_matrix_general(
    py::array_t<double, py::array::c_style | py::array::forcecast> bhat,
    py::array_t<double, py::array::c_style | py::array::forcecast> cov_stack,
    py::array_t<double, py::array::c_style | py::array::forcecast> u_stack,
    py::array_t<double, py::array::c_style | py::array::forcecast> u_factors,
    py::array_t<int, py::array::c_style | py::array::forcecast> u_ranks
) {
    auto bbuf = bhat.request();
    auto vbuf = cov_stack.request();
    auto ubuf = u_stack.request();
    auto fbuf = u_factors.request();
    auto rbuf = u_ranks.request();

    if (bbuf.ndim != 2) {
        throw std::invalid_argument("bhat must be 2D (J, R)");
    }
    if (vbuf.ndim != 2 && vbuf.ndim != 3) {
        throw std::invalid_argument("cov_stack must be 2D (R, R) or 3D (J, R, R)");
    }
    if (ubuf.ndim != 3) {
        throw std::invalid_argument("u_stack must be 3D (P, R, R)");
    }
    if (fbuf.ndim != 3) {
        throw std::invalid_argument("u_factors must be 3D (P, R, Kmax)");
    }
    if (rbuf.ndim != 1) {
        throw std::invalid_argument("u_ranks must be 1D (P)");
    }

    std::size_t J = static_cast<std::size_t>(bbuf.shape[0]);
    std::size_t R = static_cast<std::size_t>(bbuf.shape[1]);
    std::size_t P = static_cast<std::size_t>(ubuf.shape[0]);
    std::size_t Kmax = static_cast<std::size_t>(fbuf.shape[2]);
    bool common_cov = (vbuf.ndim == 2);

    if (common_cov) {
        if (vbuf.shape[0] != static_cast<py::ssize_t>(R) ||
            vbuf.shape[1] != static_cast<py::ssize_t>(R)) {
            throw std::invalid_argument("2D cov_stack must have shape (R, R)");
        }
    } else if (vbuf.shape[0] != static_cast<py::ssize_t>(J) ||
               vbuf.shape[1] != static_cast<py::ssize_t>(R) ||
               vbuf.shape[2] != static_cast<py::ssize_t>(R)) {
        throw std::invalid_argument("3D cov_stack must have shape (J, R, R)");
    }
    if (ubuf.shape[1] != static_cast<py::ssize_t>(R) ||
        ubuf.shape[2] != static_cast<py::ssize_t>(R)) {
        throw std::invalid_argument("u_stack must have shape (P, R, R)");
    }
    if (fbuf.shape[0] != static_cast<py::ssize_t>(P) ||
        fbuf.shape[1] != static_cast<py::ssize_t>(R)) {
        throw std::invalid_argument("u_factors must have shape (P, R, Kmax)");
    }
    if (rbuf.shape[0] != static_cast<py::ssize_t>(P)) {
        throw std::invalid_argument("u_ranks must have shape (P,)");
    }

    auto b = bhat.unchecked<2>();
    auto u = u_stack.unchecked<3>();
    auto f = u_factors.unchecked<3>();
    auto ranks = u_ranks.unchecked<1>();

    std::vector<double> out_buf(J * P, -std::numeric_limits<double>::infinity());

    std::vector<double> logdet_v(J, -std::numeric_limits<double>::infinity());
    std::vector<std::vector<double>> vinv_y(J, std::vector<double>(R, 0.0));
    std::vector<double> quad_v(J, 0.0);
    std::vector<std::vector<double>> y_store(J, std::vector<double>(R, 0.0));
    std::vector<std::vector<double>> chol_v;
    std::vector<std::vector<double>> v_store;
    std::vector<double> common_v;
    std::vector<double> common_chol_v;
    double common_logdet_v = -std::numeric_limits<double>::infinity();

    if (common_cov) {
        auto v = cov_stack.unchecked<2>();
        common_v.assign(R * R, 0.0);
        for (std::size_t rr = 0; rr < R; ++rr) {
            for (std::size_t cc = 0; cc < R; ++cc) {
                common_v[rr * R + cc] = v(rr, cc);
            }
        }
        if (cholesky_with_jitter(common_v, common_chol_v, R)) {
            double ld = 0.0;
            for (std::size_t rr = 0; rr < R; ++rr) {
                ld += std::log(common_chol_v[rr * R + rr]);
            }
            common_logdet_v = 2.0 * ld;
        }
        for (std::size_t j = 0; j < J; ++j) {
            for (std::size_t rr = 0; rr < R; ++rr) {
                y_store[j][rr] = b(j, rr);
            }
            if (!std::isfinite(common_logdet_v)) {
                continue;
            }
            vinv_y[j] = solve_spd_vector(common_chol_v, y_store[j], R);
            double q = 0.0;
            for (std::size_t rr = 0; rr < R; ++rr) {
                q += y_store[j][rr] * vinv_y[j][rr];
            }
            quad_v[j] = q;
            logdet_v[j] = common_logdet_v;
        }
    } else {
        auto v = cov_stack.unchecked<3>();
        chol_v.assign(J, std::vector<double>(R * R, 0.0));
        v_store.assign(J, std::vector<double>(R * R, 0.0));
        for (std::size_t j = 0; j < J; ++j) {
            std::vector<double> vj(R * R, 0.0);
            for (std::size_t rr = 0; rr < R; ++rr) {
                for (std::size_t cc = 0; cc < R; ++cc) {
                    vj[rr * R + cc] = v(j, rr, cc);
                }
                y_store[j][rr] = b(j, rr);
            }
            v_store[j] = vj;
            std::vector<double> l;
            if (!cholesky_with_jitter(vj, l, R)) {
                continue;
            }
            chol_v[j] = std::move(l);

            double ld = 0.0;
            for (std::size_t rr = 0; rr < R; ++rr) {
                ld += std::log(chol_v[j][rr * R + rr]);
            }
            logdet_v[j] = 2.0 * ld;

            vinv_y[j] = solve_spd_vector(chol_v[j], y_store[j], R);
            double q = 0.0;
            for (std::size_t rr = 0; rr < R; ++rr) {
                q += y_store[j][rr] * vinv_y[j][rr];
            }
            quad_v[j] = q;
        }
    }

    for (std::size_t p = 0; p < P; ++p) {
        int rank = ranks(p);
        bool use_woodbury = rank >= 0;
        std::size_t r = (rank <= 0) ? 0 : static_cast<std::size_t>(rank);
        if (r > Kmax) {
            use_woodbury = false;
        }

        std::vector<double> u_factor;
        if (use_woodbury && r > 0) {
            u_factor.assign(R * r, 0.0);
            for (std::size_t rr = 0; rr < R; ++rr) {
                for (std::size_t cc = 0; cc < r; ++cc) {
                    u_factor[rr * r + cc] = f(p, rr, cc);
                }
            }
        }

        std::vector<double> up(R * R, 0.0);
        if (!use_woodbury) {
            for (std::size_t rr = 0; rr < R; ++rr) {
                for (std::size_t cc = 0; cc < R; ++cc) {
                    up[rr * R + cc] = u(p, rr, cc);
                }
            }
        }

        if (common_cov) {
            if (!std::isfinite(common_logdet_v)) {
                continue;
            }
            if (use_woodbury) {
                if (r == 0) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (J > 512)
#endif
                    for (std::int64_t jj = 0; jj < static_cast<std::int64_t>(J); ++jj) {
                        std::size_t j = static_cast<std::size_t>(jj);
                        out_buf[j * P + p] = -0.5 * (static_cast<double>(R) * kLog2Pi + common_logdet_v + quad_v[j]);
                    }
                    continue;
                }

                std::vector<double> vinv_u = solve_spd_matrix_cols(common_chol_v, u_factor, R, r);
                std::vector<double> m(r * r, 0.0);
                for (std::size_t i = 0; i < r; ++i) {
                    for (std::size_t k = 0; k < r; ++k) {
                        double sum = (i == k) ? 1.0 : 0.0;
                        for (std::size_t rr = 0; rr < R; ++rr) {
                            sum += u_factor[rr * r + i] * vinv_u[rr * r + k];
                        }
                        m[i * r + k] = sum;
                    }
                }
                std::vector<double> chol_m;
                if (!cholesky_with_jitter(m, chol_m, r)) {
                    continue;
                }
                double logdet_m = 0.0;
                for (std::size_t i = 0; i < r; ++i) {
                    logdet_m += std::log(chol_m[i * r + i]);
                }
                logdet_m *= 2.0;
                double total_logdet = common_logdet_v + logdet_m;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (J > 512)
#endif
                for (std::int64_t jj = 0; jj < static_cast<std::int64_t>(J); ++jj) {
                    std::size_t j = static_cast<std::size_t>(jj);
                    std::vector<double> btmp(r, 0.0);
                    for (std::size_t i = 0; i < r; ++i) {
                        double sum = 0.0;
                        for (std::size_t rr = 0; rr < R; ++rr) {
                            sum += u_factor[rr * r + i] * vinv_y[j][rr];
                        }
                        btmp[i] = sum;
                    }
                    auto minv_b = solve_spd_vector(chol_m, btmp, r);
                    double correction = 0.0;
                    for (std::size_t i = 0; i < r; ++i) {
                        correction += btmp[i] * minv_b[i];
                    }
                    double quad = quad_v[j] - correction;
                    out_buf[j * P + p] = -0.5 * (static_cast<double>(R) * kLog2Pi + total_logdet + quad);
                }
                continue;
            }

            std::vector<double> sigma(R * R, 0.0);
            for (std::size_t idx = 0; idx < R * R; ++idx) {
                sigma[idx] = common_v[idx] + up[idx];
            }
            std::vector<double> l;
            if (!cholesky_with_jitter(sigma, l, R)) {
                continue;
            }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (J > 512)
#endif
            for (std::int64_t jj = 0; jj < static_cast<std::int64_t>(J); ++jj) {
                std::size_t j = static_cast<std::size_t>(jj);
                out_buf[j * P + p] = log_mvn_from_chol_with_obs(l, y_store[j], R);
            }
            continue;
        }

        if (use_woodbury) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (J > 512)
#endif
            for (std::int64_t jj = 0; jj < static_cast<std::int64_t>(J); ++jj) {
                std::size_t j = static_cast<std::size_t>(jj);
                if (!std::isfinite(logdet_v[j])) {
                    out_buf[j * P + p] = -std::numeric_limits<double>::infinity();
                    continue;
                }
                if (r == 0) {
                    out_buf[j * P + p] = -0.5 * (static_cast<double>(R) * kLog2Pi + logdet_v[j] + quad_v[j]);
                    continue;
                }
                bool ok = true;
                double ll = log_mvn_cov_plus_u_woodbury(
                    chol_v[j],
                    logdet_v[j],
                    quad_v[j],
                    vinv_y[j],
                    u_factor,
                    R,
                    r,
                    ok
                );
                out_buf[j * P + p] = ok ? ll : -std::numeric_limits<double>::infinity();
            }
            continue;
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (J > 512)
#endif
        for (std::int64_t jj = 0; jj < static_cast<std::int64_t>(J); ++jj) {
            std::size_t j = static_cast<std::size_t>(jj);
            if (!std::isfinite(logdet_v[j])) {
                out_buf[j * P + p] = -std::numeric_limits<double>::infinity();
                continue;
            }

            std::vector<double> sigma(R * R, 0.0);
            for (std::size_t idx = 0; idx < R * R; ++idx) {
                sigma[idx] = v_store[j][idx] + up[idx];
            }
            std::vector<double> l;
            if (!cholesky_with_jitter(sigma, l, R)) {
                out_buf[j * P + p] = -std::numeric_limits<double>::infinity();
                continue;
            }
            out_buf[j * P + p] = log_mvn_from_chol_with_obs(l, y_store[j], R);
        }
    }

    py::array_t<double> out({static_cast<py::ssize_t>(J), static_cast<py::ssize_t>(P)});
    auto out_m = out.mutable_unchecked<2>();
    for (std::size_t j = 0; j < J; ++j) {
        for (std::size_t p = 0; p < P; ++p) {
            out_m(j, p) = out_buf[j * P + p];
        }
    }

    return out;
}

py::dict compute_posterior_general_stats(
    py::array_t<double, py::array::c_style | py::array::forcecast> bhat,
    py::array_t<double, py::array::c_style | py::array::forcecast> shat_alpha,
    py::array_t<double, py::array::c_style | py::array::forcecast> cov_stack,
    py::array_t<double, py::array::c_style | py::array::forcecast> u_stack,
    py::array_t<double, py::array::c_style | py::array::forcecast> u_factors,
    py::array_t<int, py::array::c_style | py::array::forcecast> u_ranks,
    py::array_t<double, py::array::c_style | py::array::forcecast> posterior_weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> A,
    bool output_posterior_cov = false
) {
    auto bbuf = bhat.request();
    auto sabuf = shat_alpha.request();
    auto vbuf = cov_stack.request();
    auto ubuf = u_stack.request();
    auto fbuf = u_factors.request();
    auto rbuf = u_ranks.request();
    auto wbuf = posterior_weights.request();
    auto abuf = A.request();

    if (bbuf.ndim != 2 || sabuf.ndim != 2) {
        throw std::invalid_argument("bhat and shat_alpha must be 2D");
    }
    if (vbuf.ndim != 3 || ubuf.ndim != 3) {
        throw std::invalid_argument("cov_stack and u_stack must be 3D");
    }
    if (fbuf.ndim != 3 || rbuf.ndim != 1) {
        throw std::invalid_argument("u_factors must be 3D and u_ranks must be 1D");
    }
    if (wbuf.ndim != 2 || abuf.ndim != 2) {
        throw std::invalid_argument("posterior_weights and A must be 2D");
    }

    std::size_t J = static_cast<std::size_t>(bbuf.shape[0]);
    std::size_t R = static_cast<std::size_t>(bbuf.shape[1]);
    std::size_t P = static_cast<std::size_t>(ubuf.shape[0]);
    std::size_t Q = static_cast<std::size_t>(abuf.shape[0]);
    std::size_t Kmax = static_cast<std::size_t>(fbuf.shape[2]);

    if (sabuf.shape[0] != static_cast<py::ssize_t>(J) || sabuf.shape[1] != static_cast<py::ssize_t>(R)) {
        throw std::invalid_argument("shat_alpha must match bhat shape");
    }
    if (vbuf.shape[0] != static_cast<py::ssize_t>(J) ||
        vbuf.shape[1] != static_cast<py::ssize_t>(R) ||
        vbuf.shape[2] != static_cast<py::ssize_t>(R)) {
        throw std::invalid_argument("cov_stack must have shape (J, R, R)");
    }
    if (ubuf.shape[1] != static_cast<py::ssize_t>(R) || ubuf.shape[2] != static_cast<py::ssize_t>(R)) {
        throw std::invalid_argument("u_stack must have shape (P, R, R)");
    }
    if (fbuf.shape[0] != static_cast<py::ssize_t>(P) || fbuf.shape[1] != static_cast<py::ssize_t>(R)) {
        throw std::invalid_argument("u_factors must have shape (P, R, Kmax)");
    }
    if (rbuf.shape[0] != static_cast<py::ssize_t>(P)) {
        throw std::invalid_argument("u_ranks must have shape (P,)");
    }
    if (wbuf.shape[0] != static_cast<py::ssize_t>(J) || wbuf.shape[1] != static_cast<py::ssize_t>(P)) {
        throw std::invalid_argument("posterior_weights must have shape (J, P)");
    }
    if (abuf.shape[1] != static_cast<py::ssize_t>(R)) {
        throw std::invalid_argument("A must have shape (Q, R)");
    }

    auto b = bhat.unchecked<2>();
    auto sa = shat_alpha.unchecked<2>();
    auto v = cov_stack.unchecked<3>();
    auto u = u_stack.unchecked<3>();
    auto f = u_factors.unchecked<3>();
    auto ranks = u_ranks.unchecked<1>();
    auto w = posterior_weights.unchecked<2>();
    auto a = A.unchecked<2>();

    std::vector<double> res_mean(J * Q, 0.0);
    std::vector<double> res_mean2(J * Q, 0.0);
    std::vector<double> res_zero(J * Q, 0.0);
    std::vector<double> res_neg(J * Q, 0.0);
    std::vector<double> post_sec_w_sum;
    if (output_posterior_cov) {
        post_sec_w_sum.assign(J * Q * Q, 0.0);
    }

    std::vector<std::vector<double>> v_store(J, std::vector<double>(R * R, 0.0));
    std::vector<std::vector<double>> y_store(J, std::vector<double>(R, 0.0));
    std::vector<std::vector<double>> chol_v(J, std::vector<double>(R * R, 0.0));
    std::vector<std::vector<double>> vinv_bhat(J, std::vector<double>(R, 0.0));
    std::vector<std::uint8_t> chol_ok(J, 0);
    std::vector<double> AS(J * Q * R, 0.0);
    const double tiny = std::numeric_limits<double>::min();
    constexpr double kWeightEps = 1e-14;

    for (std::size_t j = 0; j < J; ++j) {
        std::vector<double> vj(R * R, 0.0);
        for (std::size_t r = 0; r < R; ++r) {
            for (std::size_t c = 0; c < R; ++c) {
                vj[r * R + c] = v(j, r, c);
            }
            y_store[j][r] = b(j, r);
        }
        v_store[j] = vj;
        std::vector<double> chol;
        if (cholesky_with_jitter(vj, chol, R)) {
            chol_ok[j] = 1;
            chol_v[j] = std::move(chol);
            vinv_bhat[j] = solve_spd_vector(chol_v[j], y_store[j], R);
        }

        for (std::size_t q = 0; q < Q; ++q) {
            for (std::size_t r = 0; r < R; ++r) {
                AS[(j * Q + q) * R + r] = a(q, r) * sa(j, r);
            }
        }
    }

    for (std::size_t p = 0; p < P; ++p) {
        int rank = ranks(p);
        bool use_woodbury = rank >= 0;
        std::size_t r = (rank <= 0) ? 0 : static_cast<std::size_t>(rank);
        if (r > Kmax) {
            use_woodbury = false;
        }

        std::vector<double> U(R * R, 0.0);
        std::vector<double> L;
        if (use_woodbury && r > 0) {
            L.assign(R * r, 0.0);
            for (std::size_t rr = 0; rr < R; ++rr) {
                for (std::size_t cc = 0; cc < r; ++cc) {
                    L[rr * r + cc] = f(p, rr, cc);
                }
            }
        } else {
            for (std::size_t rr = 0; rr < R; ++rr) {
                for (std::size_t cc = 0; cc < R; ++cc) {
                    U[rr * R + cc] = u(p, rr, cc);
                }
            }
        }

        if (use_woodbury) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (J > 512)
#endif
            for (std::int64_t jj = 0; jj < static_cast<std::int64_t>(J); ++jj) {
                std::size_t j = static_cast<std::size_t>(jj);
                double wjp = w(j, p);
                if (wjp <= kWeightEps || !chol_ok[j]) {
                    continue;
                }
                std::vector<double> col_tmp;
                std::vector<double> y_tmp;
                std::vector<double> x_tmp;
                std::vector<double> zq;
                std::vector<double> xq;

                const double* as_j = &AS[(j * Q) * R];

                if (r == 0) {
                    for (std::size_t q = 0; q < Q; ++q) {
                        res_zero[j * Q + q] += wjp;
                    }
                    continue;
                }

                std::vector<double> vinvL;
                solve_spd_matrix_cols_inplace(chol_v[j], L, R, r, vinvL, col_tmp, y_tmp, x_tmp);
                std::vector<double> S(r * r, 0.0);
                for (std::size_t i = 0; i < r; ++i) {
                    for (std::size_t k = 0; k < r; ++k) {
                        double sum = (i == k) ? 1.0 : 0.0;
                        for (std::size_t rr = 0; rr < R; ++rr) {
                            sum += L[rr * r + i] * vinvL[rr * r + k];
                        }
                        S[i * r + k] = sum;
                    }
                }
                std::vector<double> cholS;
                if (!cholesky_with_jitter(S, cholS, r)) {
                    continue;
                }

                std::vector<double> g(r, 0.0);
                for (std::size_t i = 0; i < r; ++i) {
                    double sum = 0.0;
                    for (std::size_t rr = 0; rr < R; ++rr) {
                        sum += L[rr * r + i] * vinv_bhat[j][rr];
                    }
                    g[i] = sum;
                }
                std::vector<double> h;
                solve_spd_vector_inplace(cholS, g, r, y_tmp, h);

                std::vector<double> Z(Q * r, 0.0);  // Z = AS_j @ L
                for (std::size_t q = 0; q < Q; ++q) {
                    for (std::size_t i = 0; i < r; ++i) {
                        double sum = 0.0;
                        for (std::size_t rr = 0; rr < R; ++rr) {
                            sum += as_j[q * R + rr] * L[rr * r + i];
                        }
                        Z[q * r + i] = sum;
                    }
                }

                std::vector<double> K;
                if (output_posterior_cov) {
                    std::vector<double> Ir(r * r, 0.0);
                    for (std::size_t i = 0; i < r; ++i) {
                        Ir[i * r + i] = 1.0;
                    }
                    solve_spd_matrix_inplace(cholS, Ir, r, K, col_tmp, y_tmp, x_tmp);
                }

                for (std::size_t q = 0; q < Q; ++q) {
                    double mu = 0.0;
                    for (std::size_t i = 0; i < r; ++i) {
                        mu += Z[q * r + i] * h[i];
                    }
                    if (!std::isfinite(mu)) {
                        mu = 0.0;
                    }

                    double vq = 0.0;
                    if (output_posterior_cov) {
                        for (std::size_t i = 0; i < r; ++i) {
                            double kzi = 0.0;
                            for (std::size_t kk = 0; kk < r; ++kk) {
                                kzi += K[i * r + kk] * Z[q * r + kk];
                            }
                            vq += Z[q * r + i] * kzi;
                        }
                    } else {
                        zq.assign(r, 0.0);
                        for (std::size_t i = 0; i < r; ++i) {
                            zq[i] = Z[q * r + i];
                        }
                        solve_spd_vector_inplace(cholS, zq, r, y_tmp, xq);
                        for (std::size_t i = 0; i < r; ++i) {
                            vq += zq[i] * xq[i];
                        }
                    }
                    if (!std::isfinite(vq) || vq < 0.0) {
                        vq = 0.0;
                    }

                    std::size_t idx = j * Q + q;
                    res_mean[idx] += wjp * mu;
                    res_mean2[idx] += wjp * (mu * mu + vq);
                    if (vq == 0.0) {
                        res_zero[idx] += wjp;
                    } else {
                        double sd = std::sqrt(std::max(vq, tiny));
                        double neg = ndtr_scalar(-mu / sd);
                        if (!std::isfinite(neg)) {
                            neg = 0.0;
                        }
                        res_neg[idx] += wjp * neg;
                    }
                }

                if (output_posterior_cov) {
                    for (std::size_t q = 0; q < Q; ++q) {
                        for (std::size_t qq = 0; qq < Q; ++qq) {
                            double pvar = 0.0;
                            for (std::size_t i = 0; i < r; ++i) {
                                double kz = 0.0;
                                for (std::size_t kk = 0; kk < r; ++kk) {
                                    kz += K[i * r + kk] * Z[qq * r + kk];
                                }
                                pvar += Z[q * r + i] * kz;
                            }
                            std::size_t idx = (j * Q + q) * Q + qq;
                            double curr_muq = 0.0;
                            double curr_muqq = 0.0;
                            for (std::size_t i = 0; i < r; ++i) {
                                curr_muq += Z[q * r + i] * h[i];
                                curr_muqq += Z[qq * r + i] * h[i];
                            }
                            post_sec_w_sum[idx] += wjp * (pvar + curr_muq * curr_muqq);
                        }
                    }
                }
            }
            continue;
        }

        // Full-rank fallback: use Cholesky solves on T = V_j + U_p (no matrix inversion).
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (J > 512)
#endif
        for (std::int64_t jj = 0; jj < static_cast<std::int64_t>(J); ++jj) {
            std::size_t j = static_cast<std::size_t>(jj);
            double wjp = w(j, p);
            if (wjp <= kWeightEps || !chol_ok[j]) {
                continue;
            }
            std::vector<double> col_tmp;
            std::vector<double> y_tmp;
            std::vector<double> x_tmp;

            std::vector<double> T(R * R, 0.0);
            for (std::size_t idx = 0; idx < R * R; ++idx) {
                T[idx] = v_store[j][idx] + U[idx];
            }
            std::vector<double> cholT;
            if (!cholesky_with_jitter(T, cholT, R)) {
                continue;
            }

            std::vector<double> t_inv_b;
            solve_spd_vector_inplace(cholT, y_store[j], R, y_tmp, t_inv_b);
            std::vector<double> mu1(R, 0.0);
            for (std::size_t rr = 0; rr < R; ++rr) {
                double sum = 0.0;
                for (std::size_t cc = 0; cc < R; ++cc) {
                    sum += U[rr * R + cc] * t_inv_b[cc];
                }
                mu1[rr] = sum;
            }

            std::vector<double> X;  // X = T^{-1} U
            solve_spd_matrix_inplace(cholT, U, R, X, col_tmp, y_tmp, x_tmp);
            std::vector<double> U1(R * R, 0.0);
            for (std::size_t rr = 0; rr < R; ++rr) {
                for (std::size_t cc = 0; cc < R; ++cc) {
                    double sum = 0.0;
                    for (std::size_t kk = 0; kk < R; ++kk) {
                        sum += U[rr * R + kk] * X[kk * R + cc];
                    }
                    U1[rr * R + cc] = U[rr * R + cc] - sum;
                }
            }

            const double* as_j = &AS[(j * Q) * R];
            std::vector<double> muA(Q, 0.0);
            std::vector<double> temp(Q * R, 0.0);
            std::vector<double> post_var(Q, 0.0);

            for (std::size_t q = 0; q < Q; ++q) {
                double m = 0.0;
                for (std::size_t rr = 0; rr < R; ++rr) {
                    m += mu1[rr] * as_j[q * R + rr];
                }
                muA[q] = std::isfinite(m) ? m : 0.0;

                for (std::size_t rr = 0; rr < R; ++rr) {
                    double sum = 0.0;
                    for (std::size_t cc = 0; cc < R; ++cc) {
                        sum += as_j[q * R + cc] * U1[cc * R + rr];
                    }
                    temp[q * R + rr] = sum;
                }

                double vq = 0.0;
                for (std::size_t rr = 0; rr < R; ++rr) {
                    vq += temp[q * R + rr] * as_j[q * R + rr];
                }
                if (!std::isfinite(vq) || vq < 0.0) {
                    vq = 0.0;
                }
                post_var[q] = vq;
            }

            for (std::size_t q = 0; q < Q; ++q) {
                std::size_t idx = j * Q + q;
                double m = muA[q];
                double vq = post_var[q];
                res_mean[idx] += wjp * m;
                res_mean2[idx] += wjp * (m * m + vq);
                if (vq == 0.0) {
                    res_zero[idx] += wjp;
                } else {
                    double sd = std::sqrt(std::max(vq, tiny));
                    double neg = ndtr_scalar(-m / sd);
                    if (!std::isfinite(neg)) {
                        neg = 0.0;
                    }
                    res_neg[idx] += wjp * neg;
                }
            }

            if (output_posterior_cov) {
                for (std::size_t q = 0; q < Q; ++q) {
                    for (std::size_t qq = 0; qq < Q; ++qq) {
                        double pvar = 0.0;
                        for (std::size_t rr = 0; rr < R; ++rr) {
                            pvar += temp[q * R + rr] * as_j[qq * R + rr];
                        }
                        std::size_t idx = (j * Q + q) * Q + qq;
                        post_sec_w_sum[idx] += wjp * (pvar + muA[q] * muA[qq]);
                    }
                }
            }
        }
    }

    py::array_t<double> mean_out({static_cast<py::ssize_t>(J), static_cast<py::ssize_t>(Q)});
    py::array_t<double> mean2_out({static_cast<py::ssize_t>(J), static_cast<py::ssize_t>(Q)});
    py::array_t<double> zero_out({static_cast<py::ssize_t>(J), static_cast<py::ssize_t>(Q)});
    py::array_t<double> neg_out({static_cast<py::ssize_t>(J), static_cast<py::ssize_t>(Q)});
    auto mean_m = mean_out.mutable_unchecked<2>();
    auto mean2_m = mean2_out.mutable_unchecked<2>();
    auto zero_m = zero_out.mutable_unchecked<2>();
    auto neg_m = neg_out.mutable_unchecked<2>();

    for (std::size_t j = 0; j < J; ++j) {
        for (std::size_t q = 0; q < Q; ++q) {
            std::size_t idx = j * Q + q;
            mean_m(j, q) = res_mean[idx];
            mean2_m(j, q) = res_mean2[idx];
            zero_m(j, q) = res_zero[idx];
            neg_m(j, q) = res_neg[idx];
        }
    }

    py::dict out;
    out["mean"] = mean_out;
    out["mean2"] = mean2_out;
    out["zero"] = zero_out;
    out["neg"] = neg_out;

    if (output_posterior_cov) {
        py::array_t<double> sec_out(
            {static_cast<py::ssize_t>(J), static_cast<py::ssize_t>(Q), static_cast<py::ssize_t>(Q)}
        );
        auto sec_m = sec_out.mutable_unchecked<3>();
        for (std::size_t j = 0; j < J; ++j) {
            for (std::size_t q = 0; q < Q; ++q) {
                for (std::size_t qq = 0; qq < Q; ++qq) {
                    sec_m(j, q, qq) = post_sec_w_sum[(j * Q + q) * Q + qq];
                }
            }
        }
        out["post_sec_w_sum"] = sec_out;
    } else {
        out["post_sec_w_sum"] = py::none();
    }

    return out;
}

py::dict compute_posterior_general_stats_from_loglik(
    py::array_t<double, py::array::c_style | py::array::forcecast> bhat,
    py::array_t<double, py::array::c_style | py::array::forcecast> shat_alpha,
    py::array_t<double, py::array::c_style | py::array::forcecast> cov_stack,
    py::array_t<double, py::array::c_style | py::array::forcecast> u_stack,
    py::array_t<double, py::array::c_style | py::array::forcecast> u_factors,
    py::array_t<int, py::array::c_style | py::array::forcecast> u_ranks,
    py::array_t<double, py::array::c_style | py::array::forcecast> component_loglik,
    py::array_t<double, py::array::c_style | py::array::forcecast> component_pi,
    py::array_t<double, py::array::c_style | py::array::forcecast> A,
    bool output_posterior_cov = false
) {
    auto bbuf = bhat.request();
    auto ubuf = u_stack.request();
    auto llbuf = component_loglik.request();
    auto pbuf = component_pi.request();

    if (bbuf.ndim != 2) {
        throw std::invalid_argument("bhat must be 2D");
    }
    if (ubuf.ndim != 3) {
        throw std::invalid_argument("u_stack must be 3D");
    }
    if (llbuf.ndim != 2 || pbuf.ndim != 1) {
        throw std::invalid_argument("component_loglik must be 2D and component_pi must be 1D");
    }

    std::size_t J = static_cast<std::size_t>(bbuf.shape[0]);
    std::size_t P = static_cast<std::size_t>(ubuf.shape[0]);
    if (llbuf.shape[0] != static_cast<py::ssize_t>(J) || llbuf.shape[1] != static_cast<py::ssize_t>(P)) {
        throw std::invalid_argument("component_loglik must have shape (J, P)");
    }
    if (pbuf.shape[0] != static_cast<py::ssize_t>(P)) {
        throw std::invalid_argument("component_pi must have shape (P,)");
    }

    auto ll = component_loglik.unchecked<2>();
    auto pi = component_pi.unchecked<1>();
    py::array_t<double> posterior_weights({static_cast<py::ssize_t>(J), static_cast<py::ssize_t>(P)});
    auto w = posterior_weights.mutable_unchecked<2>();

    std::vector<double> log_pi(P, -std::numeric_limits<double>::infinity());
    for (std::size_t p = 0; p < P; ++p) {
        double pip = pi(p);
        if (pip > 0.0 && std::isfinite(pip)) {
            log_pi[p] = std::log(pip);
        }
    }

    const double tiny = std::numeric_limits<double>::min();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (J > 512)
#endif
    for (std::int64_t jj = 0; jj < static_cast<std::int64_t>(J); ++jj) {
        std::size_t j = static_cast<std::size_t>(jj);
        double row_max = -std::numeric_limits<double>::infinity();
        for (std::size_t p = 0; p < P; ++p) {
            double lp = log_pi[p];
            if (!std::isfinite(lp)) {
                continue;
            }
            double llik = ll(j, p);
            if (!std::isfinite(llik)) {
                continue;
            }
            double v = lp + llik;
            if (v > row_max) {
                row_max = v;
            }
        }

        if (!std::isfinite(row_max)) {
            for (std::size_t p = 0; p < P; ++p) {
                w(j, p) = 0.0;
            }
            continue;
        }

        double sumexp = 0.0;
        for (std::size_t p = 0; p < P; ++p) {
            double lp = log_pi[p];
            if (!std::isfinite(lp)) {
                continue;
            }
            double llik = ll(j, p);
            if (!std::isfinite(llik)) {
                continue;
            }
            sumexp += std::exp(lp + llik - row_max);
        }
        if (!(sumexp > tiny) || !std::isfinite(sumexp)) {
            for (std::size_t p = 0; p < P; ++p) {
                w(j, p) = 0.0;
            }
            continue;
        }
        double inv = 1.0 / sumexp;

        for (std::size_t p = 0; p < P; ++p) {
            double lp = log_pi[p];
            if (!std::isfinite(lp)) {
                w(j, p) = 0.0;
                continue;
            }
            double llik = ll(j, p);
            if (!std::isfinite(llik)) {
                w(j, p) = 0.0;
                continue;
            }
            w(j, p) = std::exp(lp + llik - row_max) * inv;
        }
    }

    return compute_posterior_general_stats(
        bhat,
        shat_alpha,
        cov_stack,
        u_stack,
        u_factors,
        u_ranks,
        posterior_weights,
        A,
        output_posterior_cov
    );
}

PYBIND11_MODULE(_edcpp, m) {
    m.doc() = "Internal C++ backend for extreme deconvolution in pymash";

    m.def(
        "fit_extreme_deconvolution",
        &fit_extreme_deconvolution,
        py::arg("ydata"),
        py::arg("ycovar"),
        py::arg("xamp"),
        py::arg("xcovar"),
        py::arg("maxiter") = 500,
        py::arg("tol") = 1e-6,
        py::arg("w") = 0.0,
        py::arg("verbose") = false
    );

    m.def(
        "calc_lik_matrix_general",
        &calc_lik_matrix_general,
        py::arg("bhat"),
        py::arg("cov_stack"),
        py::arg("u_stack"),
        py::arg("u_factors"),
        py::arg("u_ranks")
    );

    m.def(
        "compute_posterior_general_stats",
        &compute_posterior_general_stats,
        py::arg("bhat"),
        py::arg("shat_alpha"),
        py::arg("cov_stack"),
        py::arg("u_stack"),
        py::arg("u_factors"),
        py::arg("u_ranks"),
        py::arg("posterior_weights"),
        py::arg("A"),
        py::arg("output_posterior_cov") = false
    );

    m.def(
        "compute_posterior_general_stats_from_loglik",
        &compute_posterior_general_stats_from_loglik,
        py::arg("bhat"),
        py::arg("shat_alpha"),
        py::arg("cov_stack"),
        py::arg("u_stack"),
        py::arg("u_factors"),
        py::arg("u_ranks"),
        py::arg("component_loglik"),
        py::arg("component_pi"),
        py::arg("A"),
        py::arg("output_posterior_cov") = false
    );

    m.def(
        "openmp_enabled",
        []() {
#ifdef _OPENMP
            return true;
#else
            return false;
#endif
        }
    );
}
