#include "hcl-c/SharedLib/HCLRuntimeUtils.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// reference:
// https://github.com/llvm/llvm-project/blob/bd672e2fc03823e536866da6721b9f053cfd586b/mlir/lib/ExecutionEngine/CRunnerUtils.cpp#L59

#define MAX_LINE_LENGTH 4096

template <typename T> void loadMemref(int64_t rank, void *ptr, char *str) {
  // Open the input file
  FILE *fp = fopen(str, "r");
  if (!fp) {
    perror("Error opening file");
    return;
  }

  // Read the file line by line
  char line[MAX_LINE_LENGTH];
  T *array = NULL;
  int array_size = 0;
  while (fgets(line, MAX_LINE_LENGTH, fp)) {
    // Parse the line and add the values to the array
    char *token = strtok(line, ",");
    while (token) {
      // Resize the array if necessary
      if (array_size % 8 == 0) {
        array = (T *)realloc(array, (array_size + 8) * sizeof(T));
      }
      // Convert the token to a float and add it to the array
      array[array_size++] = atof(token);
      // Get the next token
      token = strtok(NULL, ",");
    }
  }

  // Close the file
  fclose(fp);

  // Print the array
  printf("LoadMemref: array loaded from file (%d elements):\n", array_size);
  for (int i = 0; i < array_size; i++) {
    std::cout << array[i] << " ";
  }
  printf("\n");

  // Copy data from array to memref buffer
  UnrankedMemRefType<T> unranked_memref = {rank, ptr};
  DynamicMemRefType<T> memref(unranked_memref);
  memcpy(memref.data, array, array_size * sizeof(T));

  // Free the array
  free(array);
}

extern "C" void loadMemrefI32(int64_t rank, void *ptr, char *str) {
  loadMemref<int32_t>(rank, ptr, str);
}

extern "C" void loadMemrefI64(int64_t rank, void *ptr, char *str) {
  loadMemref<int64_t>(rank, ptr, str);
}

extern "C" void loadMemrefF32(int64_t rank, void *ptr, char *str) {
  loadMemref<float>(rank, ptr, str);
}

extern "C" void loadMemrefF64(int64_t rank, void *ptr, char *str) {
  loadMemref<double>(rank, ptr, str);
}
