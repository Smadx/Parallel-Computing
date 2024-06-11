# 矩阵LU分解
## 算法设计
串行的Doolittle算法用三个部分可并行:
- 对角线以下元素的缩放
- 更新子矩阵的元素
- 提取L和U矩阵
并行算法如下:
```algorithm
Doolittle(A, N, L, U):
    Input:
        A - 待分解的矩阵
        N - 矩阵的大小
        L - 下三角矩阵
        U - 上三角矩阵
    Output:
        无直接输出，但L和U将被修改为分解后的结果

    for k=0 to N-1 do
        for i=k+1 to N-1 par-do
            A[i][k] /= A[k][k]
        end for
    end for

        for i=k+1 to N-1 par-do
            for j=k+1 to N-1 do
                A[i][j] -= A[i][k] * A[k][j]
            end for
        end for

    for i=0 to N-1 par-do
        for j=0 to N-1 do
            if i>j do
                L[i][j] = A[i][j]
                U[i][j] = 0.0
            else if i==j do
                L[i][j] = 1.0
                U[i][j] = A[i][j]
            else
                L[i][j] = 0.0
                U[i][j] = A[i][j]
            end if
        end for
    end for
```
其中并行部分负载较为均衡,额外开销主要体现在线程的创建通信和同步以及数据分割.
## 实验结果
当增大线程数时,运行时间先减小后增大,无法达到线性加速比.LU分解的外层循环未并行,后期随着线程数的增大通信和同步开销变多,因而降低了加速比.
## 结论
外层循环无法并行的算法很难达到线性加速比,随着线程数的增加,OpenMP的额外开销逐渐增大反而会降低加速比,因此对于这种算法我们应该选择一个合适的线程数.
# 单源最短路径
## 算法设计
DeltaStepping的这些部分可并行:
- 找最小桶
- 松弛操作
我们有下面的算法:
```algorithm
DeltaStepping(G, V, src, delta=10):
    Input:
        G - 图的邻接表表示
        V - 顶点数
        src - 源点
        delta - 桶内极差
    Output:
        dist - 从源点到每个顶点的最短距离

    Init:
        dist设置为INF
        V个空桶B[V]
        把src加入第一个桶

    Parallel:
        while 有不为空的桶 do
            Single:
                找到第一个不为空的桶B[i]
            Prallel:
                while !B[i] do
                    Req 是B[i]中的轻边
                    R = B[i]
                    清空B[i]
                    对Req中的轻边做松弛操作
                    更新桶
                Req 是B[i]中的重边
                松弛Req
                更新桶
```
## 实验结果
delta值对性能有很大影响,我们发现delta=10时效率较高,并行负载的均衡程度和数据分布有关,而当线程数增加时运行时间也增加,这是因为这个算法的通信开销很大.
## 总结
这个并行算法的性能可能不如串行算法,因为阻塞区和通信开销很大
# K-means聚类
## 算法设计
串行Kmeans算法的这些部分可并行:
- 分配数据点到最近的聚类中心
- 计算新的聚类中心
- 计算所有点到其聚类中心的距离之和
我们给出并行算法的核心部分L
```algorithm
assign_clusters(data, centroids):
    Input:
        data - 数据集（包含多个点）
        centroids - 聚类中心
    Output:
        C - 每个聚类中心对应的数据点集合

    初始化一个空的聚类字典 C
    为每个聚类中心初始化一个空集合

    对于数据集中的每个点:
        计算该点到所有聚类中心的距离
        将该点分配到距离最近的聚类中心对应的集合中

    返回聚类字典 C

compute_centroids(data, C, centroids, K):
    Input:
        data - 数据集
        C - 每个聚类中心对应的数据点集合
        centroids - 聚类中心
        K - 聚类中心的数量
    Output:
        无直接输出，但centroids将被更新为新的聚类中心

    初始化一个大小为 K 的点列表，每个点的坐标初始化为 0
    初始化一个大小为 K 的计数列表，初始值为 0

    对于每个聚类中心及其对应的点集合:
        将点集合中的每个点的坐标累加到对应的聚类中心的坐标上
        更新计数列表

    对于每个聚类中心:
        如果对应的计数大于 0:
            将聚类中心的坐标除以计数，得到新的聚类中心

calculate_local_total_distance(centroids, C, rank, size):
    Input:
        centroids - 聚类中心
        C - 每个聚类中心对应的数据点集合
        rank - 当前进程的rank
        size - 总进程数
    Output:
        local_total_distance - 当前进程计算的局部总距离

    初始化局部总距离为 0

    每个进程处理一些聚类:
        对于当前进程负责的每个聚类中心:
            计算聚类中心到其对应的点集合中每个点的距离之和

    返回局部总距离
```
## 实验结果
由于数据量比较小,并行额外开销大于计算开销,因此加速比会随线程数的增大而减小,这个并行负载均衡程度也和数据分布有关.
## 结论
Kmeans算法比较好并行,但是要体现并行算法性能的优势需要规模更大的数据
# 稀疏矩阵乘法
## 算法设计
CSC的spmm算法遍历稀疏矩阵的每个元素可并行,我们给出下面的算法:
```algorithm
__global__ void spmm(dense, sparse, result, pitch, M, N, P, K):
    Input:
        dense - 稠密矩阵 (M x N)
        sparse - 稀疏矩阵，以CSC格式存储 (P x (2 * K + 1))
        result - 结果矩阵 (M x P)
        pitch - 稀疏矩阵的内存跨距
        M - 稠密矩阵的行数
        N - 稠密矩阵的列数
        P - 稀疏矩阵的列数
        K - 非零元素的数量
    Output:
        无直接输出，但result将被更新为稀疏矩阵与稠密矩阵相乘的结果

    计算 row 和 col 作为线程索引
    if row < M 且 col < P:
        初始化 sum 为 0
        获取 sparse 矩阵的第 col 列的起始地址 sparse_col
        for i = 0 to sparse_col[0] - 1:
            sum += dense[row * N + sparse_col[i * 2 + 1]] * sparse_col[i * 2 + 2]
        将 sum 存储到 result[row * P + col]
```
## 实验结果
每个进程负载一个稀疏矩阵的元素,相对均衡,由于这个矩阵的规模较大,在初期我们的加速比会随线程数的增大而增大,这个算法的通信与同步开销也比较小,但还是不能达到线性加速,是一个不错的算法.
## 结论
综合以上分析，CSC的SpMM算法具有良好的强可扩放性。在固定问题规模的情况下，随着处理器数量的增加，性能提升显著，但随着核数增加，性能提升会逐渐趋于平缓，受到Amdahl定律的限制