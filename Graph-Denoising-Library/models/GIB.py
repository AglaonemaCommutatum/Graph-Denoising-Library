import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")

def load_data(path):
    data = torch.load('data\cora\Planetoid\Cora\processed\data.pt')[0]
    # print(data)

    nodes = data['x']
    edge_index = data['edge_index']

    num_nodes, num_feature = nodes.shape

    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)

    # 将边添加到邻接矩阵中
    for i in range(edge_index.shape[1]):  # 遍历每一条边
        src = edge_index[0, i]  # 源节点
        dst = edge_index[1, i]  # 目标节点
        adjacency_matrix[src, dst] = 1  # 设置邻接矩阵中的值

    return nodes.numpy(), adjacency_matrix.numpy()



def relu(x):
    return np.maximum(0, x)

class GIB_Cat:
    def __init__(self, X, A, T, k, tau=relu):
        self.X = X  # 节点特征
        self.A = A  # 邻接矩阵
        self.T = T  # 局部依赖的限制
        self.k = k  # 采样邻居的数量
        self.tau = tau  # 非线性激活函数
        self.L = 2  # 层数
        self.W = [np.random.rand(X.shape[1], X.shape[1]) for _ in range(self.L)]  # 权重初始化

    def NeighborSample(self, Z):
        sampled_neighbors = []
        for v in range(len(Z)):
            neighbors = np.where(self.A[v] > 0)[0]  # 找到邻居
            if len(neighbors) > self.k:
                sampled = np.random.choice(neighbors, self.k, replace=True)  # 采样邻居
            else:
                sampled = neighbors  # 如果邻居少于k，全部取出
            sampled_neighbors.append(sampled)
        return sampled_neighbors

    def forward(self):
        Z = self.X.copy()  # 初始化表示
        for l in range(self.L):
            Z_new = np.zeros_like(Z)
            for v in range(len(Z)):
                Z_hat = self.tau(Z[v] @ self.W[l])  # 线性变换与激活
                Z_A = self.NeighborSample(Z)  # 采样邻居

                # 过滤掉无效的邻居
                valid_neighbors = [u for u in Z_A[v] if u < len(Z_hat)]

                if valid_neighbors:  # 如果有有效邻居
                    Z_bar = np.mean([Z_hat[u] for u in valid_neighbors], axis=0)  # 聚合邻居信息
                else:
                    Z_bar = Z_hat  # 如果没有有效邻居，保持原值

                Z_new[v] = Z_bar  # 更新节点表示
            Z = Z_new  # 更新Z
        return Z  # 返回最终节点表示


class GIB_Bern:
    def __init__(self, X, A, T, tau):
        self.X = X  # 节点特征
        self.A = A  # 邻接矩阵
        self.T = T  # 局部依赖的限制
        self.tau = tau  # 非线性激活函数
        self.L = 2  # 层数
        self.W = [np.random.rand(X.shape[1], X.shape[1]) for _ in range(self.L)]  # 权重初始化

    def NeighborSample(self, Z):
        sampled_neighbors = []
        for v in range(len(Z)):
            neighbors = np.where(self.A[v] > 0)[0]  # 找到邻居
            probabilities = np.random.rand(len(neighbors))  # 生成Bernoulli概率
            sampled = neighbors[probabilities < 0.5]  # 以50%的概率采样
            sampled_neighbors.append(sampled)
        return sampled_neighbors

    def forward(self):
        Z = self.X.copy()  # 初始化表示
        for l in range(self.L):
            Z_new = np.zeros_like(Z)
            for v in range(len(Z)):
                Z_hat = self.tau(Z[v] @ self.W[l])  # 线性变换与激活
                Z_A = self.NeighborSample(Z)  # 采样邻居

                # 过滤掉无效的邻居
                valid_neighbors = [u for u in Z_A[v] if u < len(Z_hat)]

                if valid_neighbors:  # 如果有有效邻居
                    Z_bar = np.mean([Z_hat[u] for u in valid_neighbors], axis=0)  # 聚合邻居信息
                else:
                    Z_bar = Z_hat  # 如果没有有效邻居，保持原值

                Z_new[v] = Z_bar  # 更新节点表示
            Z = Z_new  # 更新Z
        return Z  # 返回最终节点表示


# 示例用法
if __name__ == "__main__":
    # # 创建一个简单的图
    # X = np.random.rand(100, 80)  # 100个节点，每个节点80个特征
    # A = np.random.randint(0, 2, (100, 100))
    path = 'data\cora\Planetoid\Cora\processed\data.pt'
    X, A = load_data(path)

    # 实例化模型
    model_cat = GIB_Cat(X, A, T=2, k=2)
    model_bern = GIB_Bern(X, A, T=2)

    # 前向传播
    output_cat = model_cat.forward()
    output_bern = model_bern.forward()

    print("GIB-Cat输出:\n", output_cat)
    print("GIB-Bern输出:\n", output_bern)