#include <omp.h>
#include <iostream>
#include <vector>

using namespace std;

/*
 * @brief Doolittle's LU decomposition
 * @param A Matrix to decompose
 * @param N Size of the matrix
 * @param L Lower triangular matrix
 * @param U Upper triangular matrix
 */
void Doolittle(vector< vector<double> > &A,int N, vector< vector<double> > &L, vector< vector<double> > &U)
{
    int i, j, k;
    for(k=0; k<N; k++)
    {
        #pragma omp parallel for shared(A, k) private(i) schedule(static)
        for(i=k+1; i<N; i++)
            A[i][k] /= A[k][k];

        #pragma omp parallel for shared(A, k) private(i, j) schedule(static)
        for(i=k+1; i<N; i++)
            for(j=k+1; j<N; j++)
                A[i][j] -= A[i][k] * A[k][j];
    }

    // Extract L and U from A
    #pragma omp parallel for shared(L, U, A, N) private(i, j) schedule(static)
    for(i=0; i<N; i++)
    {
        for(j=0; j<N; j++)
        {
            if(i>j)
            {
                L[i][j] = A[i][j];
                U[i][j] = 0.0;
            }
            else if(i==j)
            {
                L[i][j] = 1.0;
                U[i][j] = A[i][j];
            }
            else
            {
                L[i][j] = 0.0;
                U[i][j] = A[i][j];
            }
        }
    }
}

int main()
{
    int N;
    cin >> N;
    vector< vector<double> > M(N, vector<double>(N));
    vector< vector<double> > L(N, vector<double>(N));
    vector< vector<double> > U(N, vector<double>(N));
    // Read matrix
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            cin >> M[i][j];
    // Perform Doolittle's LU decomposition
    Doolittle(M, N, L, U);
    // Print L
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << L[i][j] << " ";
        cout << endl;
    }
    // Print U
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << U[i][j] << " ";
        cout << endl;
    }
    return 0;
}