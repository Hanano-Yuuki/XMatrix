#include <iostream>
#include <random>
#include <vector>
#include <chrono>

#include "matrix.h"

int main(){
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> gen(0,1);

    size_t row=5, col=3, k=1024*1024*1;
    auto m1=types::Mat<float>(row, k);
    auto m2=types::Mat<float>(k, col);

    omp_set_num_threads(16);
    m1.fill_all([&rng, &gen]{return gen(rng);}, 16); // randomize
    m2.fill_all([&rng, &gen]{return gen(rng);}, 16); // randomize

    auto start=std::chrono::high_resolution_clock::now();
    auto m=m1*m2;
    auto end=std::chrono::high_resolution_clock::now();
    std::cout<<"cpu time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<"ms"<<std::endl;
    m.print();
    std::cout<<std::endl;

    start=std::chrono::high_resolution_clock::now();
    auto d_m1=m1.to_device();
    auto d_m2=m2.to_device();
    end=std::chrono::high_resolution_clock::now();
    std::cout<<"transfer time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<"ms"<<std::endl;


    start=std::chrono::high_resolution_clock::now();
    auto d_m=d_m1*d_m2;
    cudaDeviceSynchronize();
    end=std::chrono::high_resolution_clock::now();
    std::cout<<"gpu time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<"ms"<<std::endl;
    d_m.print();
    std::cout<<std::endl;

    return 0;
}