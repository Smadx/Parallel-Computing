#include <iostream>
#include <vector>
#include <list>
#include <set>
#include <algorithm>
#include <climits>
#include <omp.h>

using namespace std;

const int INF = INT_MAX;

struct Edge
{
    int to, weight;
};

/*
 * @brief DeltaStepping求解单源最短路径
 * @param G 图
 * @param V 顶点数
 * @param src 源点
 * @param delta 桶内极差
 */
vector<int> DeltaStepping(const vector<list<Edge>> &G, int V, int src, int delta = 10)
{
    vector<int> dist(V, INF); // 距离向量，初始化为无穷大
    dist[src] = 0;

    vector<set<int>> buckets(V); // 桶数组
    int maxBucket = 0;
    int minBucket = 0;
    int temp = 0;
    buckets[0].insert(src); // 起点放入第一个桶
    #pragma omp parallel
    {
        while (maxBucket >= minBucket) // 当最大桶索引大于等于最小桶索引时
        {
            #pragma omp barrier
            #pragma omp single
            {
                temp = maxBucket; // 记录当前最大桶索引
                for (int i = minBucket; i <= maxBucket; ++i)
                {
                    if (buckets[i].empty())
                    {
                        if (i == maxBucket)
                        {
                            minBucket = maxBucket + 1;
                        }
                        continue;
                    }
                    minBucket = i; // 更新最小桶索引
                    break;
                }
            }
            #pragma omp for schedule(static) // 并行 for 循环，静态调度
            for (int i = minBucket; i <= maxBucket; ++i)
            {
                while (!buckets[i].empty())
                {
                    int u = -1; // 当前处理的顶点
                    #pragma omp critical // 临界区
                    {
                        if (!buckets[i].empty())
                        {
                            u = *buckets[i].begin(); // 获取桶中第一个顶点
                            buckets[i].erase(buckets[i].begin()); // 从桶中移除该顶点
                        }
                    }
                    if (u == -1)
                    {
                        continue;
                    }
                    for (const auto &edge : G[u])
                    {
                        int v = edge.to;
                        int w = edge.weight;
                        int newDist = dist[u] + w; // 松弛操作
                        if (newDist < dist[v])
                        {
                            #pragma omp critical // 临界区
                            {
                                if (newDist < dist[v])
                                {
                                    if (dist[v] != INF)
                                    {
                                        int oldBucket = dist[v] / delta; // 计算旧桶索引
                                        buckets[oldBucket].erase(v); // 从旧桶中移除
                                    }
                                    dist[v] = newDist; // 更新距离
                                    int newBucket = newDist / delta; // 计算新桶索引
                                    if (newBucket >= buckets.size())
                                    {
                                        buckets.resize(newBucket + 1); // 调整桶数组大小
                                    }
                                    buckets[newBucket].insert(v); // 将顶点加入新桶
                                    temp = max(temp, newBucket); // 更新最大桶索引
                                }
                            }
                        }
                    }
                }
            }
            #pragma omp single
            {
                if (temp > maxBucket)
                {
                    maxBucket = temp; // 更新最大桶索引
                }
                else
                {
                    for (int i = maxBucket; i >= minBucket; --i)
                    {
                        if (!buckets[i].empty())
                        {
                            maxBucket = i; // 找到新的最大桶索引
                            break;
                        }
                    }
                }
            }
        }
    }

    return dist; // 返回最短路径距离向量
}

int main()
{
    int V, E, src;
    cin >> V >> E >> src; // 输入顶点数、边数和起点

    vector<list<Edge>> G(V); // 图的邻接表表示

    for (int i = 0; i < E; ++i)
    {
        int u, v, w;
        cin >> u >> v >> w;
        G[u].push_back({v, w});
        G[v].push_back({u, w});
    }

    omp_set_num_threads(8);

    vector<int> dist = DeltaStepping(G, V, src);

    for (int i = 0; i < V; ++i)
    {
        if (dist[i] == INF)
            cout << "INF ";
        else
            cout << dist[i] << " ";
    }

    return 0;
}