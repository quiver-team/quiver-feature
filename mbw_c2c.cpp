#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstring>

using namespace std;

int64_t test(char *src, char *dst, long *index, int index_size, int dim) {
    auto start = chrono::steady_clock::now();
    size_t cnt = dim * 4;
    for (int i = 0; i < index_size; i++) {
        char *src_beg = src + cnt * index[i];
        char *dst_beg = dst + cnt * i;
        memcpy(dst_beg, src_beg, cnt);
    }
    auto end = chrono::steady_clock::now();
    return chrono::duration_cast<chrono::microseconds>(end - start).count();
}

int main() {
    int mode = 0;
    int total_size = 50000000;
    int index_size = 1000000;
    int dim = 128;
    vector<float> src((size_t)total_size * dim, 0);
    vector<float> dst((size_t)index_size * dim, 0);
    vector<long> index(total_size, 0);
    iota(index.begin(), index.end(), 0);
    random_shuffle(index.begin(), index.end());
    test((char*)src.data(), (char*)dst.data(), index.data(), index_size, dim);
    int64_t cost = 0;
    int iter = 10;
    int64_t total_bytes = (int64_t)index_size * iter * dim * 4;
    for (int i = 0; i < iter; i++) {
        cost += test((char*)src.data(), (char*)dst.data(), index.data(), index_size, dim);
    }
    cout << total_bytes * 1.0 / cost << "MB/s "  << total_bytes << " bytes " << cost << " us"<< endl;
}