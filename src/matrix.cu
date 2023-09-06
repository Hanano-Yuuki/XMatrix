#include <matrix.h>

namespace ops {

template <class T>
__global__ void mtxAdd(int n, int m, T *alpha, T *lhs, T *beta, T *rhs,
                       T *out) {
  int size = n * m;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = (*alpha) * lhs[idx] + (*beta) * rhs[idx];
  }
}

template <class T>
__global__ void mtxMulti(int n, int m, int k, T *alpha, T *lhs, T *rhs, T *beta,
                         T *out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * m) {
    int row = idx / m;
    int col = idx % m;
    T sum = 0;
    for (int i = 0; i < k; i++) {
      sum += lhs[row * k + i] * rhs[i * m + col];
    }
    out[idx] = (*beta) * out[idx] + (*alpha) * sum;
  }
}

template <class T>
void mtxMulti_cublas(cublasHandle_t &handle, cublasOperation_t transa,
                     cublasOperation_t transb, int n, int m, int k,
                     const T *alpha, const T *lhs, int lda, const T *rhs,
                     int ldb, const T *beta, T *out, int ldc) {
  if constexpr (std::is_same<float, T>::value) {
    cublasSgemm_v2(handle, transa, transb, n, m, k, alpha, lhs, n, rhs, k, beta,
                   out, n);
  } else if constexpr (std::is_same<double, T>::value) {
    cublasDgemm_v2(handle, transa, transb, n, m, k, alpha, lhs, n, rhs, k, beta,
                   out, n);
  } else {
    throw std::runtime_error(
        "cublas gemm error: unsupported type, use mtxMulti instead.");
  }
}

}  // namespace ops

namespace types {

template <typename T>
Mat<T>::Mat(size_t n, size_t m, HWType type,
            const std::initializer_list<T> initL)
    : n(n), m(m), type(type) {
  if (n * m == 0) return;  // create a null matrix as placeholder
  if (initL.size() > n * m) {
    std::clog << "[warning]: initial list is too long, may drop tail elements."
              << std::endl;
  }
  if (type == HWType::HOST) {
    data = (T *)malloc(n * m * sizeof(T));
    if (initL.size()) {
      memcpy(data, initL.begin(), n * m * sizeof(T));
    }
  } else if (type == HWType::DEVICE) {
    cudaMalloc(&cudata, n * m * sizeof(T));
    if (n * m) {
      cudaMemcpy(cudata, initL.begin(), n * m * sizeof(T),
                 cudaMemcpyHostToDevice);
    }
  }
}

template <class T>
Mat<T>::Mat(const Mat<T> &rhs) {
  n = rhs.n;
  m = rhs.m;
  type = rhs.type;
  if (type == HWType::HOST) {
    data = (T *)malloc(n * m * sizeof(T));
    memcpy(data, rhs.data, n * m * sizeof(T));
  } else if (type == HWType::DEVICE) {
    cudaMalloc(&cudata, n * m * sizeof(T));
    cudaMemcpy(cudata, rhs.cudata, n * m * sizeof(T), cudaMemcpyDeviceToDevice);
  }
}

template <class T>
Mat<T>::Mat(Mat<T> &&rhs) {
  std::swap(n, rhs.n);
  std::swap(m, rhs.m);
  std::swap(type, rhs.type);
  if (type == HWType::HOST) {
    std::swap(data, rhs.data);
  } else if (type == HWType::DEVICE) {
    std::swap(cudata, rhs.cudata);
  }
}

template <class T>
Mat<T>::~Mat() {
  if (type == HWType::HOST) {
    free(data);
  } else if (type == HWType::DEVICE) {
    cudaFree(cudata);
  }
}

template <class T>
Mat<T> &Mat<T>::operator=(const Mat<T> &rhs) {
  n = rhs.n;
  m = rhs.m;
  type = rhs.type;
  if (type == HWType::HOST) {
    data = (T *)realloc(data, n * m * sizeof(T));
    memcpy(data, rhs.data, n * m * sizeof(T));
  } else if (type == HWType::DEVICE) {
    std::clog << "[warning]: copy device matrix, may cause performance issue."
              << std::endl;
    cudaDeviceSynchronize();
    cudaFree(cudata);
    cudaMalloc(&cudata, n * m * sizeof(T));
    cudaMemcpy(cudata, rhs.cudata, n * m * sizeof(T), cudaMemcpyDeviceToDevice);
  }
  return *this;
}

template <class T>
Mat<T> &Mat<T>::operator=(Mat<T> &&rhs) {
  std::swap(n, rhs.n);
  std::swap(m, rhs.m);
  std::swap(type, rhs.type);
  if (type == HWType::HOST) {
    std::swap(data, rhs.data);
  } else if (type == HWType::DEVICE) {
    std::swap(cudata, rhs.cudata);
  }
  return *this;
}

template <class T>
T &Mat<T>::operator[](const size_t idx) {
  assert(idx < n * m);
  if (type == HWType::HOST) {
    return data[idx];
  } else {
    throw std::runtime_error("index error: cannot access device matrix.");
  }
}

template <class T>
T &Mat<T>::operator()(const size_t i, const size_t j) {
  assert(i < n && j < m);
  return this->operator[](i * m + j);
}

template <class T>
Mat<T> Mat<T>::operator*(const Mat<T> &rhs) {
  if (m != rhs.n) {
    throw std::runtime_error("matrix multiplication error: shape mismatch.");
  }
  if (type != rhs.type) {
    throw std::runtime_error(
        "matrix multiplication error: type mismatch, cannot multiply a host "
        "matrix with a device matrix.");
  }
  Mat<T> ret(n, rhs.m, type, std::initializer_list<T>{});
  if (type == HWType::HOST) {
#pragma omp parallel for schedule(dynamic)
    for (size_t idx = 0; idx < n * rhs.m; idx++) {
      size_t i = idx / rhs.m;
      size_t j = idx - i * rhs.m;
      T sum = 0;
      for (int k = 0; k < m; k++) {
        sum += data[i * m + k] * rhs.data[k * rhs.m + j];
      }
      ret.data[i * rhs.m + j] = sum;
    }
  } else if (type == HWType::DEVICE) {
    size_t threadsPerBlock = 32;
    size_t blocksPerGrid = (n * rhs.m + threadsPerBlock - 1) / threadsPerBlock;
    ops::mtxMulti<<<blocksPerGrid, threadsPerBlock>>>(n, rhs.m, m, cudata,
                                                      rhs.cudata, ret.cudata);
    cudaDeviceSynchronize();
  }
  return ret;
}

template <class T>
void Mat<T>::set(const std::initializer_list<T> &argsList) {
  size_t cpysz = std::min(argsList.size(), n * m);
  if (type == HWType::HOST) {
    memcpy(data, argsList.begin(), cpysz * sizeof(T));
  } else if (type == HWType::DEVICE) {
    cudaDeviceSynchronize();
    cudaMemcpy(cudata, argsList.begin(), cpysz * sizeof(T),
               cudaMemcpyHostToDevice);
  }
  if (argsList.size() > m * n)
    std::clog << "[warning]: argument list is too long, may drop tail elements."
              << std::endl;
}

template <class T>
size_t Mat<T>::row() const {
  return n;
}

template <class T>
size_t Mat<T>::col() const {
  return m;
}

template <class T>
void Mat<T>::reshape(ssize_t new_n, ssize_t new_m) {
  if (new_n > 0 && new_m > 0) {
    // pass
  } else if (new_n > 0 && new_m == -1) {
    new_m = (n * m + new_n - 1) / new_n;
  } else if (new_n == -1 && new_m > 0) {
    new_n = (n * m + new_m - 1) / new_m;
  } else {
    throw std::runtime_error("reshape error: new shape is invalid.");
  }
  resize(new_n, new_m);
}

template <class T>
void Mat<T>::fill_all(std::function<T()> gen, int tnum) {
  if (type == HWType::DEVICE)
    throw std::runtime_error("fill_all error: cannot fill device matrix.");
#pragma omp parallel for schedule(dynamic) num_threads(tnum)
  for (size_t i = 0; i < n * m; i++) {
    data[i] = gen();
  }
}

template <class T>
void Mat<T>::print() {
  if (type == HWType::HOST) {
    std::cout << "HOST matrix:" << std::endl;
  } else if (type == HWType::DEVICE) {
    std::cout << "DEVICE matrix:" << std::endl;
    cudaDeviceSynchronize();
  }
  if (type == HWType::HOST) {
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < m; j++) {
        std::cout << data[i * m + j] << ' ';
      }
      std::cout << std::endl;
    }
  } else if (type == HWType::DEVICE) {
    cudaDeviceSynchronize();
    std::unique_ptr<T[]> tmp(new T[n * m]);
    cudaMemcpy(tmp.get(), cudata, n * m * sizeof(T), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < m; j++) {
        std::cout << tmp[i * m + j] << ' ';
      }
      std::cout << std::endl;
    }
  }
}

template <class T>
void Mat<T>::resize(size_t new_n, size_t new_m) {
  if (new_n * new_m < n * m) {
    std::clog
        << "[warning]: resize matrix to smaller size, may drop tail elements."
        << std::endl;
  }
  if (type == HWType::HOST) {
    data = (T *)realloc(data, new_n * new_m * sizeof(T));
    if (new_n * new_m > n * m) {
      memset(data + n * m, 0, (new_n * new_m - n * m) * sizeof(T));
    }
  } else if (type == HWType::DEVICE) {
    size_t cpysz = std::min(new_n * new_m, n * m);
    cudaDeviceSynchronize();
    T *new_cudata;
    cudaMalloc(&new_cudata, new_n * new_m * sizeof(T));
    cudaMemcpy(new_cudata, cudata, cpysz * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaFree(cudata);
    cudata = new_cudata;
  }
  n = new_n;
  m = new_m;
}

template <class T>
Mat<T> Mat<T>::to_device() {
  Mat<T> ret(n, m, HWType::DEVICE, std::initializer_list<T>{});
  if (type == HWType::HOST) {
    cudaMemcpy(ret.cudata, data, n * m * sizeof(T), cudaMemcpyHostToDevice);
  } else if (type == HWType::DEVICE) {
    cudaDeviceSynchronize();
    cudaMemcpy(ret.cudata, cudata, n * m * sizeof(T), cudaMemcpyDeviceToDevice);
  }
  return ret;
}

template <class T>
Mat<T> Mat<T>::to_host() {
  Mat<T> ret(n, m, HWType::HOST, std::initializer_list<T>{});
  if (type == HWType::HOST) {
    memcpy(ret.data, data, n * m * sizeof(T));
  } else if (type == HWType::DEVICE) {
    cudaDeviceSynchronize();
    cudaMemcpy(ret.data, cudata, n * m * sizeof(T), cudaMemcpyDeviceToHost);
  }
  return ret;
}

template class Mat<int>;
template class Mat<float>;

}  // namespace types