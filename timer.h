#include <time.h>
#include <chrono>
#ifndef TIMER_H
#define TIMER_H
#define ITERATIONS 20
// kernel time
std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;
// cublas create
std::chrono::high_resolution_clock::time_point s_create;
std::chrono::high_resolution_clock::time_point e_create;
// data transfer
std::chrono::high_resolution_clock::time_point s_data;
std::chrono::high_resolution_clock::time_point e_data;

// kernel time
uint64_t s_compute_cycles;
uint64_t e_compute_cycles;
// cublas create
uint64_t s_create_cycles;
uint64_t e_create_cycles;
// data transfer
uint64_t s_data_cycles;
uint64_t e_data_cycles;


uint64_t rdtsc() {
  unsigned int lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
}

#endif
