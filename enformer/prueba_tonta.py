import tensorflow as tf
import enformer
import numpy as np
import tensorflow as tf
import enformer
import numpy as np
import sonnet as snt

model = enformer.Enformer(channels=1536,
                          num_heads=8,
                          num_transformer_layers=11,
                          pooling_type='max')

extended_model = enformer.ExtendedEnformer(model)

input_tensor = tf.random.uniform((1, 196608, 4), dtype=tf.float32)
output_1 = model(input_tensor, is_training=True)
output_2 = extended_model(input_tensor, is_training=True)


def count_parameters(model):
    return np.sum([np.prod(var.shape) for var in model.trainable_variables])

print(model.variables)
#print(count_parameters(model))
#print(count_parameters(extended_model))

# 242 865 452 o1
# 221 492 737 o2

"""
Trunk output shape: (1, 1536, 768) con 1536 / 4 canales
Trunk output shape: (1, 1536, 3072) con 1536 canales
242 865 452 (o1) -> 221 489 664 (solo la parte de trunk)
221 489 664 + 3072 * 1 + 1 = 221 492 737 Confirmamos o2

Sin embargo estamos perdiendo millones de pesos por el camino.
221 489 664 * 5313 * 3072 + 5313 + 1643 * 3072 + 1643 = 242 865 452 Confirmamos o1
21 375 788 Neuronas que perdemos
Sólo perdemos 16 326 849 neuronas útiles (las de humano) ya que el conocimiento que
aportaba el ratón al humano (supuestamente) está contenido en trunk, no en su cabeza.
242 865 452 (o1) - 221 492 737 (o2) = 21 372 715 (21 375 788 + 3073) Son neuronas perdidas + neuronas implementadas
"""