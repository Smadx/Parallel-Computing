#include <mpi.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <iostream>
#include <climits>
#include <random>
#include <iomanip>
#include <cfloat>
#include <functional>

using namespace std;

class Point
{
public:
    vector<double> coordinates;

    Point()
    {
        coordinates = vector<double>(0);
    }

    Point(int M)
    {
        coordinates.resize(M);
    }

    Point(int size, double value)
    {
        coordinates.resize(size, value);
    }

    double distance(const Point& other) const
    {
        double dist = 0.0;
        for (int i = 0; i < coordinates.size(); ++i)
        {
            dist += pow(coordinates[i] - other.coordinates[i], 2);
        }
        return sqrt(dist);
    }

    Point& operator+=(const Point& other)
    {
        if (coordinates.size() != other.coordinates.size()) {
            throw std::length_error("Vectors must be the same length.");
        }
        for (int i = 0; i < coordinates.size(); ++i)
        {
            coordinates[i] += other.coordinates[i];
        }
        return *this;
    }

    Point operator/(int n) const
    {
        Point result(coordinates.size());
        for (int i = 0; i < coordinates.size(); ++i)
        {
            result.coordinates[i] = coordinates[i] / n;
        }
        return result;
    }

    double& operator[](int index)
    {
        return coordinates[index];
    }

    const double* data() const
    {
        return coordinates.data();
    }

    double* data()
    {
        return coordinates.data();
    }

    bool operator<(const Point& other) const
    {
        return coordinates < other.coordinates;
    }

    bool operator==(const Point& other) const
    {
        return coordinates == other.coordinates;
    }
};

// 自定义哈希函数
struct PointHash
{
    size_t operator()(const Point& point) const
    {
        size_t hash = 0;
        for (double coord : point.coordinates)
        {
            hash ^= std::hash<double>()(coord);
        }
        return hash;
    }
};

/*
 * @brief 把data中的每个点分配到最近的聚类中心
 * @param data 数据集
 * @param centroids 聚类中心
 */
unordered_map<int, unordered_set<Point, PointHash>> assign_clusters(const vector<Point>& data, const vector<Point>& centroids)
{
    unordered_map<int, unordered_set<Point, PointHash>> C;
    for (int i = 0; i < centroids.size(); ++i)
    {
        C[i] = unordered_set<Point, PointHash>();
    }

    for (int i = 0; i < data.size(); ++i)
    {
        double min_dist = DBL_MAX;
        int cluster = -1;
        for (int j = 0; j < centroids.size(); ++j)
        {
            double dist = data[i].distance(centroids[j]);
            if (dist < min_dist)
            {
                min_dist = dist;
                cluster = j;
            }
        }
        C[cluster].insert(data[i]);
    }

    return C;
}

/*
 * @brief 计算新的聚类中心
 * @param data 数据集
 * @param C 聚类结果
 * @param centroids 新的聚类中心
 * @param K 聚类中心的个数
 */
void compute_centroids(const vector<Point>& data, const unordered_map<int, unordered_set<Point, PointHash>>& C, vector<Point>& centroids, int K)
{
    centroids.assign(K, Point(data[0].coordinates.size(), 0.0));
    vector<int> counts(K, 0);

    for (const auto& pair : C)
    {
        int cluster_id = pair.first;
        const unordered_set<Point, PointHash>& cluster_points = pair.second;

        for (const Point& p : cluster_points)
        {
            centroids[cluster_id] += p;
            counts[cluster_id]++;
        }

        if (counts[cluster_id] > 0)
        {
            centroids[cluster_id] = centroids[cluster_id] / counts[cluster_id];
        }
    }
}

/*
 * @brief 计算所有点到其聚类中心的距离之和
 * @param centroids 聚类中心
 * @param C 聚类结果
 * @param rank 当前进程的rank
 * @param size 总进程数
 */
double calculate_local_total_distance(const vector<Point>& centroids, const unordered_map<int, unordered_set<Point, PointHash>>& C, int rank, int size)
{
    double local_total_distance = 0.0;

    // 每个进程处理一些聚类
    for (int cluster_id = rank; cluster_id < centroids.size(); cluster_id += size)
    {
        const auto& cluster_points = C.at(cluster_id);
        for (const Point& p : cluster_points)
        {
            local_total_distance += p.distance(centroids[cluster_id]);
        }
    }

    return local_total_distance;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N, M, K;

    vector<Point> data;
    vector<Point> centroids;
    unordered_map<int, unordered_set<Point, PointHash>> C;

    if (rank == 0)
    {
        // 读取数据
        cin >> N >> M >> K;
        data.resize(N, Point(M));
        centroids.resize(K, Point(M));

        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < M; ++j)
            {
                cin >> data[i][j];
            }
        }

        // 随机初始化K个聚类中心
        default_random_engine generator;
        uniform_int_distribution<int> distribution(0, N - 1);
        for (int i = 0; i < K; ++i)
        {
            centroids[i] = data[distribution(generator)];
        }
    }

    // 广播数据
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 初始化数据
    if (rank != 0)
    {
        data.resize(N, Point(M));
        centroids.resize(K, Point(M));
    }

    // 向所有进程广播数据
    for (int i = 0; i < N; ++i)
    {
        MPI_Bcast(data[i].data(), M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // 向所有进程广播聚类中心
    for (int i = 0; i < K; ++i)
    {
        MPI_Bcast(centroids[i].data(), M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int max_iters = 100;

    for (int iter = 0; iter < max_iters; ++iter)
    {
        // 把数据点划到最近的聚类中心
        C = assign_clusters(data, centroids);

        // 计算新的聚类中心
        compute_centroids(data, C, centroids, K);

        // 广播新的聚类中心
        for (int i = 0; i < K; ++i)
        {
            MPI_Bcast(centroids[i].data(), M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    // 计算局部总距离
    double local_total_distance = calculate_local_total_distance(centroids, C, rank, size);

    // 汇总全局总距离
    double global_total_distance = 0.0;
    MPI_Reduce(&local_total_distance, &global_total_distance, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << fixed << setprecision(2) << global_total_distance << endl;
    }

    MPI_Finalize();
    return 0;
}
