import tensorflow as tf

class SAM():
    def __init__(self, base_optimizer, rho=0.005, eps=1e-12):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        self.rho = rho
        self.eps = eps
        self.base_optimizer = base_optimizer

    def first_step(self, gradients, trainable_vars):
        self.e_ws = []
        grad_norm = tf.linalg.global_norm(gradients)
        ew_multiplier = self.rho / (grad_norm + self.eps)
        for i in range(len(trainable_vars)):
            e_w = tf.math.multiply(gradients[i], ew_multiplier)
            trainable_vars[i].assign_add(e_w)
            self.e_ws.append(e_w)

    def second_step(self, gradients, trainable_variables):
        for i in range(len(trainable_variables)):
            trainable_variables[i].assign_add(-self.e_ws[i])
        # do the actual "sharpness-aware" update
        self.base_optimizer.apply_gradients(zip(gradients, trainable_variables))
