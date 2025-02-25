import torch

x = torch.randn(1, 1, 10, requires_grad=True)
w = torch.randn(1, 1, 3)
s = 2
p = 1

y = torch.nn.functional.conv1d(x, w, stride=s, padding=p)
v = torch.rand_like(y, requires_grad=False)

# calculate the gradient by autograd
loss = torch.sum(v * y)
loss.backward()

# calculate the gradient manually by transposed convolution
with torch.no_grad():
    # output_padding = x.shape[-1] - s * (y.shape[-1] - 1) + 2 * p - w.shape[-1]
    # it equals to 1 in this case
    output_padding = 1
    u = torch.nn.functional.conv_transpose1d(
        v, w, stride=s, padding=p, output_padding=output_padding
    )

# compare these two results
assert torch.allclose(x.grad, u)
print("Test passed.")
