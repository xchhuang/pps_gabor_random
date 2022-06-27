import torch
from sklearn.neighbors import NearestNeighbors


def knn(x1, x2, k):
    # print(x1.shape, x2.shape)
    inner = -2 * torch.matmul(x2.transpose(2, 1), x1)
    xx1 = torch.sum(x1 ** 2, dim=1, keepdim=True)
    xx2 = torch.sum(x2 ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx1 - inner - xx2.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    # print(idx.shape)
    return idx


def get_neighbor_torch(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # print(feature.shape)
    g = feature - x
    x_diff = g[:, :, :, 0]
    y_diff = g[:, :, :, 1]
    neighbor_dist = torch.sqrt(x_diff ** 2 + y_diff ** 2)
    # print('neighbor_dist:', neighbor_dist.shape)
    avg_neighbor_dist = torch.mean(neighbor_dist, -1)
    # print(g.shape, avg_neighbor_dist.shape)
    # neighbor_dist =
    # feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return avg_neighbor_dist


def get_knearest_neighbors(p, D, num_neighbors):
    if torch.cuda.is_available():
        X = p[:, :D].detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
    else:
        X = p[:, :D].detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
    return torch.from_numpy(distances)[:, 1:], torch.from_numpy(indices)[:, 1:]
