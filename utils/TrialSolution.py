import tensorflow as tf
import numpy as np

class ODETrialSolution(tf.keras.models.Model):
    def __init__(self, conditions, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 activation='sigmoid', 
                 call_method=None):
        super(ODETrialSolution, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Boundary conditions
        self.conditions = conditions
        self.call_method = call_method

        self.hidden_layer = tf.keras.layers.Dense(units=self.hidden_size, activation=activation)
        self.output_layer = tf.keras.layers.Dense(units=self.output_size, activation='linear')

    def call(self, X):
        X = tf.convert_to_tensor(X)
        if not self.call_method is None:
            return self.call_method(self, X)
        response = self.hidden_layer(X)
        response = self.output_layer(response)

        boundary_value = tf.constant(0., dtype='float64', shape=response.get_shape())

        for condition in self.conditions:
            vanishing = tf.constant(1., dtype='float64', shape=response.get_shape())
            temp_bc = 0
            temp_bc = tf.reshape(condition['function'](X), shape=boundary_value.shape)
            for vanisher in self.conditions:
                if vanisher['variable'] != condition['variable'] and vanisher['value'] != condition['value']:
                    vanishing *= (X[:, vanisher['variable']] - tf.constant(vanisher['value'], dtype='float64', shape=boundary_value.shape))
            boundary_value += temp_bc * vanishing
            response *= (tf.constant(condition['value'], dtype='float64', shape=boundary_value.shape) - tf.reshape(X[:, condition['variable']], shape=boundary_value.shape))
        response += boundary_value
        return response
    
class TrialSolution(tf.keras.models.Model):
    def __init__(self, conditions, input_size, hidden_size, output_size=1, activation='sigmoid'):
        super(TrialSolution, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.conditions = conditions
        self.trial_solution = ODETrialSolution(conditions=conditions, 
                                               input_size=input_size, 
                                               hidden_size=hidden_size, 
                                               output_size=output_size,
                                               activation=activation)
    def call(self, X):
        return self.trial_solution(X)

    def train(self, X, diff_loss, epochs, verbose=True, message_frequency=1, learning_rate=0.1, optimizer_name='Adam'):
        optimizer = None
        if optimizer_name == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        
        @tf.function
        def train_step(X):
            with tf.GradientTape() as tape:
                loss = diff_loss(self, X)
            gradients = tape.gradient(loss, self.trial_solution.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trial_solution.trainable_variables))
            
        for epoch in range(epochs):
            for x in X:
                x_tensor = tf.reshape(x, shape=(1, X.shape[1]))           
                train_step(x_tensor)
            if verbose and ((epoch+1) % message_frequency == 0):
                print(f'Epoch: {epoch+1} Loss: {diff_loss(self, X)}')