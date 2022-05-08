#pragma once
#include <stdio.h>
#include <stdlib.h>

#define QUIVER_FEATURE_ASSERT(B, X, ...) \
  {                                      \
    if (!(B)) {                          \
      fprintf(stdout, X, ##__VA_ARGS__); \
      fflush(stdout);                    \
      exit(-1);                          \
    }                                    \
  }
