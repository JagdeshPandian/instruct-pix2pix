import torch_fidelity
from metrics import is_fid_pytorch


wrapped_generator = torch_fidelity.GenerativeModelModuleWrapper(generator, 128, 'normal', 0)

metrics_dict = torch_fidelity.calculate_metrics(
    input1=wrapped_generator,
    input2='cifar10-train',
    cuda=True,
    isc=True,
    fid=True,
    kid=True,
    prc=True,
    verbose=False,
)
print(metrics_dict)

