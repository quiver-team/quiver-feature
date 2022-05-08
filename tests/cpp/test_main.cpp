
// Usage: ./progam -s for server and ./program for client component

#include <iostream>
#include <string>
void test_pipe(int argc, char** argv);
void test_dist_tensor_server(int argc, char** argv);

int main(int argc, char** argv) {
  test_dist_tensor_server(argc, argv);
}
