/** -----------------------
@file factorizeOnCPU.cpp
@brief Prime factorization using linear sieve algorithm
Provides the implementation of function sets for implementing CPU/QPU 
comparison benchmarks for prime factorization using the linear sieve algorithm.<br/>
Dependencies are:
	- stdc++ (C++20)
	- openmp (recomended)
	@copyright (c) CEA LIST, QuLIST team <br/>
	Distributed under EU Public Licence (EUPL) <br/>
@verbatim
@article{10.1145/359657.359660,
author = {Gries, David and Misra, Jayadev},
title = {A linear sieve algorithm for finding prime numbers},
year = {1978},
issue_date = {Dec. 1978},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {21},
number = {12},
issn = {0001-0782},
url = {https://doi.org/10.1145/359657.359660},
doi = {10.1145/359657.359660},
abstract = {A new algorithm is presented for finding all primes between 2 and n. The algorithm executes in time proportional to n (assuming that multiplication of integers not larger than n can be performed in unit time). The method has the same arithmetic complexity as the algorithm presented by Mairson [6]; however, our version is perhaps simpler and more elegant. It is also easily extended to find the prime factorization of all integers between 2 and n in time proportional to n.},
journal = {Commun. ACM},
month = dec,
pages = {999–1003},
numpages = {5}, 
keywords = {primes, data structures, algorithms}}
@endverbatim
---------------------------*/

#include <bits/stdc++.h>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace chrono;

using ll = long long;

// Returns the prime factorization of every integer in [2, n]
// as a vector of vectors: factors[k] = list of prime factors of k (with multiplicity)
vector<vector<int>> factorize_all(int n, int time_limit_sec = 60) {
    auto start = high_resolution_clock::now();

    // 1. Compute smallest prime factor (SPF) for every number up to n
    vector<int> spf(n + 1);
    vector<int> primes;
    primes.reserve(n / 10);          // rough estimate

    #pragma omp parallel
    {
        // sequential initialization is fine and avoids false sharing
        #pragma omp single
        {
            iota(spf.begin(), spf.end(), 0);
            spf[0] = spf[1] = 1;
        }
    }

    // Parallel sieve for SPF
    // We mark multiples of primes in parallel segments
    for (int i = 2; i <= n; ++i) {
        if (spf[i] == i) {               // i is prime
            primes.push_back(i);
        }
        for (int p : primes) {
            if (p > spf[i] || (ll)i * p > n) break;
            spf[i * p] = p;
        }
    }

    // Alternative more parallelizable sieve (useful for very large n)
    // Uncomment the block below and comment the sequential loop above if desired.
    /*
    #pragma omp parallel for schedule(dynamic)
    for (int i = 2; i * i <= n; ++i) {
        if (spf[i] == i) {               // i is prime
            for (ll j = (ll)i * i; j <= n; j += i) {
                if (spf[j] == j) spf[j] = i;
            }
        }
    }
    */

    auto mid = high_resolution_clock::now();
    double sieve_time = duration<double>(mid - start).count();
    if (sieve_time > time_limit_sec) {
        cerr << "Sieve exceeded time limit\n";
        return {};
    }

    // 2. Factorize every number using the SPF table
    vector<vector<int>> factors(n + 1);

    #pragma omp parallel for schedule(dynamic, 1024)
    for (int k = 2; k <= n; ++k) {
        // Check remaining time occasionally
        if ((k & 0xFFFF) == 0) {
            auto now = high_resolution_clock::now();
            if (duration<double>(now - start).count() > time_limit_sec) {
                // We cannot break cleanly from OpenMP, so just skip further work
                continue;
            }
        }

        int x = k;
        while (x > 1) {
            factors[k].push_back(spf[x]);
            x /= spf[x];
        }
    }

    // Number 1 has empty factorization (or you can put {1} if preferred)
    factors[1] = {};

    return factors;
}

// factorize_c to bridge with python3 
#include <cstdlib>          // for malloc / free

extern "C" {

int factorize_all_c(int n, int time_limit_sec,
                    int** out_data, int** out_lengths, int* out_num_rows)
{
    try {
        auto result = factorize_all(n, time_limit_sec);   // call C++ function

        *out_num_rows = static_cast<int>(result.size());
        if (*out_num_rows == 0) {
            *out_data = nullptr;
            *out_lengths = nullptr;
            return 0;
        }

        // Calculate total number of integers
        size_t total = 0;
        for (const auto& row : result)
            total += row.size();

        int* data    = static_cast<int*>(std::malloc(total * sizeof(int)));
        int* lengths = static_cast<int*>(std::malloc(*out_num_rows * sizeof(int)));

        if (!data || !lengths) {
            std::free(data);
            std::free(lengths);
            return -1;          // allocation failure
        }

        size_t idx = 0;
        for (int i = 0; i < *out_num_rows; ++i) {
            lengths[i] = static_cast<int>(result[i].size());
            for (int v : result[i])
                data[idx++] = v;
        }

        *out_data    = data;
        *out_lengths = lengths;
        return 0;               // success
    }
    catch (...) {
        return -2;              // any C++ exception
    }
}

// you need to free the allocated memory (to be called from python)
void free_factorization(int* data, int* lengths)
{
    std::free(data);
    std::free(lengths);
}

} // extern "C"

/*** Unitary test below
*/
#ifdef __UNITARY_TEST_
int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (argc > 1) {
        n = stoi(argv[1]);
    } else {
        n= 10000001;
    }

    cout << "Factorizing integers from 1 to " << n << " ...\n";

    auto start = high_resolution_clock::now();
    auto factors = factorize_all(n, 60);          // 60-second limit
    auto end = high_resolution_clock::now();

    double elapsed = duration<double>(end - start).count();
    cout << fixed << setprecision(3);
    cout << "Time: " << elapsed << " s\n";

    if (factors.empty()) {
        cout << "Computation aborted due to time limit.\n";
        return 1;
    }

    // Optional: print a few examples
    cout << "\nSample factorizations:\n";
    for (int i : {1, 2, 10, 100, 1000, n/2, n}) {
        if (i > n) continue;
        cout << i << " = ";
        if (factors[i].empty()) {
            cout << "1\n";
        } else {
            for (size_t j = 0; j < factors[i].size(); ++j) {
                if (j) cout << " * ";
                cout << factors[i][j];
            }
            cout << "\n";
        }
    }

    // If you need the full result for further processing, it is in 'factors'
    // Example: count total number of prime factors with multiplicity
    ll total = 0;
    #pragma omp parallel for reduction(+:total)
    for (int i = 1; i <= n; ++i)
        total += factors[i].size();

    cout << "\nTotal number of prime factors (with multiplicity) from 1 to "
         << n << ": " << total << "\n";

    return 0;
}
#endif

