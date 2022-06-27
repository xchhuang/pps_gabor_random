import torch
import numpy as np
import torch.nn.functional as F


class Slicing_torch(torch.nn.Module):
    def __init__(self, device, layers, repeat_rate):
        super().__init__()
        # Number of directions
        self.device = device
        self.repeat_rate = repeat_rate
        self.update_slices(layers)
        # self.target = self.compute_target(layers)

    def update_slices(self, layers):
        directions = []
        for l in layers:    # converted to [B, W, H, D]
            if l.ndim == 4:
                l = l.permute(0, 2, 3, 1)
            if l.ndim == 5:
                l = l.permute(0, 2, 3, 4, 1)

            dim_slices = l.shape[-1]
            num_slices = l.shape[-1]
            # num_slices = 512
            # print('num_slices:', num_slices)
            # print(num_slices, dim_slices)
            cur_dir = torch.randn(size=(num_slices, dim_slices)).to(self.device)
            # cur_dir = torch.from_numpy(np_random_normal)
            # print(cur_dir.shape)
            # norm = torch.sqrt(torch.sum(torch.square(cur_dir), axis=-1))
            norm = torch.sqrt(torch.sum(cur_dir ** 2, -1))

            norm = norm.view(num_slices, 1)
            cur_dir = cur_dir / norm
            # print(norm.shape)
            directions.append(cur_dir)
        # directions = torch.cat(directions)
        self.directions = directions
        target = self.compute_target(layers)
        self.target = target
        # return directions, target

    def compute_proj(self, input, layer_idx, repeat_rate):
        if input.ndim == 4:
            input = input.permute(0, 2, 3, 1)
        if input.ndim == 5:
            input = input.permute(0, 2, 3, 4, 1)

        # batch, _, _, dim = input.size()
        batch = input.size(0)
        dim = input.size(-1)
        tensor = input.view(batch, -1, dim)
        # print('before:', tensor.shape)
        tensor_permute = tensor.permute(0, 2, 1)
        # print('after:', tensor.shape, self.directions[layer_idx].shape)
        # Project each pixel feature onto directions (batch dot product)
        sliced = torch.matmul(self.directions[layer_idx], tensor_permute)
        # print('sliced(torch):', sliced.shape)
        # # Sort projections for each direction
        sliced, _ = torch.sort(sliced)
        # print('sliced(torch):', sliced.shape, self.repeat_rate)
        sliced = sliced.repeat_interleave(repeat_rate ** 2, dim=-1)
        # print('sliced(torch):', sliced.shape, self.repeat_rate)
        # print(sliced[0, 0, 0:100])
        sliced = sliced.view(batch, -1)
        return sliced

    def compute_target(self, layers):
        target = []
        for idx, l in enumerate(layers):
            # print('target:', idx, l.shape)
            sliced_l = self.compute_proj(l, idx, self.repeat_rate)
            # print('sliced_l:', sliced_l.shape)
            target.append(sliced_l.detach())
        return target

    def forward(self, input):
        loss = 0.0
        for idx, l in enumerate(input):
            # print('output:', idx, l.shape)
            cur_l = self.compute_proj(l, idx, 1)
            direction = self.directions[idx]
            # print('cur_l:', idx, cur_l.shape, self.target[idx].shape)
            # loss += F.mse_loss(cur_l, self.target[idx], reduction='sum') / (direction.shape[0] * direction.shape[1])
            loss += F.mse_loss(cur_l, self.target[idx])

            # loss += (cur_l - self.target[idx]).pow(1).sum()

        return loss


def compute_error(x1, x2):
    err = np.mean((x1 - x2) ** 2)
    return err


def test_expansion():
    upsampling_rate = 2
    s = np.random.rand(1, 32, 32, 256)
    l = np.random.rand(1, 32 * upsampling_rate, 32 * upsampling_rate, 256)
    s = s.astype(np.float32)
    l = l.astype(np.float32)

    s = torch.from_numpy(s)
    slicing_torch = Slicing_torch(layers=[s], repeat_rate=2)
    slicing_torch.compute_target([s])

    l = torch.from_numpy(l)
    # slicing_output = Slicing_torch(layers=[l], repeat_rate=1)
    loss = slicing_torch([l])
    print('loss:', loss)
    # err = compute_error(out_torch.numpy(), tar_torch.numpy())
    # # print('err:', err)


def main():
    test_expansion()


if __name__ == '__main__':
    main()
