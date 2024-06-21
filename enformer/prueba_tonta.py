import tensorflow as tf
import enformer
import numpy as np
import tensorflow as tf
import enformer
import numpy as np

model = enformer.Enformer(channels=1536,
                          num_heads=8,
                          num_transformer_layers=11,
                          pooling_type='max')

#extended_model = enformer.ExtendedEnformer(model)
extended_model = enformer.ExtendedEnformerMouse(model)

input_tensor = tf.random.uniform((1, 196608, 4), dtype=tf.float32)
output_1 = model(input_tensor, is_training=True)
output_2 = extended_model(input_tensor, is_training=True)


def count_parameters(model):
    return np.sum([np.prod(var.shape) for var in model.trainable_variables])

print(extended_model.variables)
#print(count_parameters(model))
#print(count_parameters(extended_model))

# 242 865 452 o1
# 224 132 444 o2
# 224 986 738 o3

"""
Trunk output shape: (1, 1536, 3072) con 1536 canales
242 865 452 (o1) -> 221 489 664 (solo la parte de trunk)
221 489 664 * 5313 * 3072 + 5313 + 1643 * 3072 + 1643 = 242 865 452 Confirmamos o1

224 132 444 = 221 489 664 (trunk) + 860 * 3072 + 860
224 986 738 = 221 489 664 (trunk) + 860 * 3072 + 860 + 278 * 3072 + 278
"""