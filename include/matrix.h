#include <cassert>
#include <iostream>
#include <memory>
#include <functional>
#include <vector>
#include <omp.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace ops {

// linear add two matrix in kernel
// @param n,m meta data of matrix
// @param lhs left hand side matrix: [n * m]
// @param rhs right hand side matri: [n * m]
// @param out output matrix
// @return out = alpha * lhs + beta * rhs
template <class T>
__global__ void mtxAdd(int n, int m, T *alpha, T *lhs, T *beta, T *rhs, T *out);


// square add two matrix in kernel
// @param lhs left hand side matrix: [n * k]
// @param rhs right hand side matrix: [k * m]
// @param out output matrix: [n * m]
// @param n,m,k meta data of matrix
// @return out = alpha * lhs * rhs + beta * out
template <class T>
__global__ void mtxMulti(int n, int m, int k, T *alpha, T *lhs, T *rhs, T* beta, T *out);

// cublas version of matrix multiplication:
// @param handle cublas lib context
// @param transa,transb transpose flag
// @param n,m,k meta data of matrix
// @param alpha,beta scalar
// @param lhs left hand side matrix: [n * k]
// @param rhs right hand side matrix: [k * m]
// @param out output matrix: [n * m]
// @param lda,ldb,ldc leading dimension of matrix
// @return out = alpha * trans(lhs) * trans(rhs) + beta * out
template <class T>
void mtxMulti_cublas(cublasHandle_t &handle, cublasOperation_t transa,
                     cublasOperation_t transb, int n, int m, int k,
                     const T *alpha, const T *lhs, int lda, const T *rhs,
                     int ldb, const T *beta, T *out, int ldc);

}  // namespace ops

namespace types {

enum class HWType { HOST, DEVICE };

template <class T>
class Mat {
  size_t n, m;
  HWType type;
  T *data, *cudata;

 public:
  // constructor
  // @param n,m meta data of matrix
  // @param type HOST or DEVICE
  Mat(size_t n, size_t m, HWType type = HWType::HOST,
      const std::initializer_list<T> initL = {});
  Mat(const Mat &rhs);
  Mat(Mat &&rhs);

  // destructor
  ~Mat();

  Mat &operator=(const Mat &rhs);
  Mat &operator=(Mat<T> &&rhs);

  T &operator[](const size_t idx);
  T &operator()(const size_t i, const size_t j);

  Mat operator*(const Mat &rhs);

  void set(const std::initializer_list<T> &argsList);
  size_t row() const;
  size_t col() const;

  void reshape(ssize_t new_n, ssize_t new_m);
  void fill_all(std::function<T()> gen,int tnum=1);

  // to_device copies a new instance, may cause performance issue
  Mat to_device();
  // to_host copies a new instance, may cause performance issue
  Mat to_host();

  void print();

 private:
  void resize(size_t new_n, size_t new_m);
};

};  // namespace types