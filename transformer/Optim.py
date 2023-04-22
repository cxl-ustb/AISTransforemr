'''A wrapper class for scheduled optimizer '''
__author__ = "Cheng XinLong"
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps, use_mlp):
        self._optimizer = optimizer
        self.lr_mul = 0.01 if use_mlp else lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        self.state_dict = self._optimizer.state_dict()
        self.use_mlp = use_mlp

    def get_lr(self):
        return self._optimizer.param_groups[0]['lr']

    def get_state_dict(self):
        return self.state_dict

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        # lr = self.lr_mul * self._get_lr_scale()
        lr = max(self.lr_mul - self.lr_mul * self.n_steps * 1.3e-4, 10e-5)
        if self.use_mlp:
            lr = max(self.lr_mul - self.lr_mul * self.n_steps * 0.5e-4, 8e-4)
        self.state_dict = self._optimizer.state_dict()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

