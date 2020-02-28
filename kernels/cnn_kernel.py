import gpytorch

from gpytorch.kernels import Kernel
from gpytorch.lazy import BatchRepeatLazyTensor, lazify

class CNNGP_Kernel(Kernel):
    def __init__(self, model, shape, **kwargs):
        # TODO: handle batch sizes
        super(CNNGP_Kernel, self).__init__(**kwargs)
        self.model = model
        self.shape = shape

    def forward(self, x1, x2=None, diag=False, last_dim_is_batch=False, **params):
        # TODO: figure out how to have multiple outputs - i believe method in 
        # def num_ouptuts_per_input -> but also for the kernel model itself
        #print('ldib', last_dim_is_batch)
        if len(x1.shape) is 3:
            x1 = x1[0]
        if x2 is not None:
            if len(x2.shape) is 3:
                x2 = x2[0]

        #print(self.batch_shape, x2 is None)        
        if len(x1.shape) is 2:
            x1 = x1.view(-1, *self.shape)
        

        # TODO: figure out how to have batched dimensional outputs
        if x2 is None:
            kernel = lazify(self.model(x1, diag=diag))
        else:
            x2 = x2.view(-1, *self.shape)
            kernel = lazify(self.model(x1, x2, diag=diag))

        #print('shape of kernel is: ', kernel.shape)
        res = BatchRepeatLazyTensor(kernel, batch_repeat=self.batch_shape)

        if last_dim_is_batch:
            res = res.permute(1, 2, 0)

        return res

    

