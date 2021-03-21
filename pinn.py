#! /usr/bin/env python

import numpy as np
from scipy.integrate import solve_ivp
import tensorflow as tf

class PINN(tf.keras.models.Sequential):
    def __init__(self, x0, f, observations=None, l=0.5, layers=None, name=None):
        super(PINN, self).__init__(layers=layers, name=name)
        self.x  = observations
        self.x0 = x0
        self.f  = f
        self.l  = tf.constant(l, shape=(1,))

    def compile(self, **kwargs):
        if 'metrics' in kwargs:
            raise TypeError("got an unexpected keyword argument 'metrics'")
        return super(PINN, self).compile(**kwargs)

    def call(self, t, training=None, mask=None):
        return self.x0 + t*super(PINN, self).call(t, training=training, mask=mask)

    def call_with_gradient(self, t, training=None, mask=None):
        with tf.GradientTape(persistent=True) as gt:
            gt.watch(t)
            y  = self.call(t, training=training, mask=mask)
            yl = [y[:,i] for i in range(y.shape[1])]
        dy = tf.stack([gt.gradient(yl[i], t)[:,0] for i in range(y.shape[1])], 1)
        return y, dy

    def train_step(self, data):
        t = data

        with tf.GradientTape() as tape:
            tape.watch(t)
            y, dy = self.call_with_gradient(t, training=True)
            fy    = self.f(t, y)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss_ode = self.compiled_loss(dy, fy, regularization_losses=self.losses)
            # Add empirical loss if available
            if self.x is not None:
                y_pred = self.call(self.x[0], training=True)
                loss_empirical = self.compiled_loss(self.x[1], y_pred, regularization_losses=self.losses)
                loss = self.l*loss_ode + (1.-self.l)*loss_empirical
            else:
                loss = loss_ode

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {"loss": loss}