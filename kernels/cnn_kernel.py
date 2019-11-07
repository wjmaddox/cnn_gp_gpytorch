import gpytorch

from gpytorch.kernels import Kernel

class CNNGP_Kernel(Kernel):
    def __init__(self, model, shape, **kwargs):
        # TODO: handle batch sizes
        super(CNNGP_Kernel, self).__init__(**kwargs)
        self.model = model
        self.shape = shape

    def forward(self, x1, x2=None, diag=False, last_dim_is_batch=False, **params):
        # TODO: figure out how to have multiple outputs - i believe method in 
        # def num_ouptuts_per_input -> but also for the kernel model itself
        
        if len(x1.shape) is 2:
            x1 = x1.view(-1, *self.shape)

        # TODO: figure out how to have batched dimensional outputs
        if x2 is None:
            return self.model(x1)
        else:
            x2 = x2.view(-1, *self.shape)

        return self.model(x1, x2, diag=diag)

    

