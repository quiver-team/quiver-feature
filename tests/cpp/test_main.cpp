
// Usage: ./progam -s for server and ./program for client component

#include <iostream>
#include <string>
void test_pipe(int argc, char** argv);
void test_dist_tensor_server(int argc, char** argv);
void test_dist_tensor_client(int argc, char** argv);
int main(int argc, char** argv) {
  int test_case = 0;
  switch (argv[1][0]) {
    case '0': {
      test_case = 0;
      break;
    }
    case '1': {
      test_case = 1;
      break;
    }
    case '2': {
      test_case = 2;
      break;
    }
  }

  ++argv;
  --argc;

  if (test_case == 0) {
    printf("Testing Pipe ...\n");
    test_pipe(argc, argv);
  } else if (test_case == 1) {
    printf("Testing DistTensorClient ...\n");
    test_dist_tensor_client(argc, argv);
  } else if (test_case == 2) {
    printf("Testing DistTensorServer ...\n");
    test_dist_tensor_server(argc, argv);
  }
}
