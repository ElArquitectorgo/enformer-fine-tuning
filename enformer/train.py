import tensorflow as tf
import sonnet as snt
from tqdm import tqdm
import numpy as np
import os
import enformer
from dataset import get_dataset
from train_utils import evaluate_model_R2, evaluate_model_PearsonR, plot_losses, get_last_epoch

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def create_step_function(model, optimizer):

  @tf.function
  def train_step(batch, optimizer_clip_norm_global=0.2):
    with tf.GradientTape() as tape:
      outputs = model(batch['sequence'], is_training=True)
      loss = tf.reduce_mean(
          tf.keras.losses.poisson(batch['target'], outputs))

    gradients = tape.gradient(loss, model.trainable_variables)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, optimizer_clip_norm_global)
    optimizer.apply(clipped_gradients, model.trainable_variables)

    return loss
  return train_step

learning_rate = tf.Variable(0., trainable=False, name='learning_rate')
optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
num_warmup_steps = 10000
target_learning_rate = 0.0005

model = enformer.Enformer(channels=1536,
                          num_heads=8,
                          num_transformer_layers=11,
                          pooling_type='max')

checkpoint = tf.train.Checkpoint(module=model)
latest = tf.train.latest_checkpoint('official_checkpoint')
checkpoint.restore(latest)
extended_model = enformer.ExtendedEnformer(model)

train_step = create_step_function(extended_model, optimizer)
config = 'Adam-50'
steps_per_epoch = 50
num_epochs = 200

human_dataset = get_dataset('human', 'train').batch(1).repeat()
validation_dataset = get_dataset('human', 'valid').batch(1).prefetch(2)

data_it = iter(human_dataset)
global_step = 0

output_dir = 'learning_curves'
os.makedirs(output_dir, exist_ok=True)

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint = tf.train.Checkpoint(module=extended_model)
latest_epoch = 0
"""latest = tf.train.latest_checkpoint(checkpoint_dir)
latest_epoch = 0

if latest is not None:
  print(f'Checkpoint found! - {latest}')
  checkpoint.restore(latest)
  latest_epoch = get_last_epoch(latest) + 1 # The loop starts at index 0 so we need to increase by one to match de real epoch
else:
  print('No checkpoints found')"""

human_losses = []
val_metrics = []
patience = 20
best_val_metric = -np.inf
patience_counter = 0

for epoch_i in range(num_epochs):
    epoch_human_losses = []
    for i in tqdm(range(steps_per_epoch)):
        global_step += 1

        if global_step > 1:
            learning_rate_frac = tf.math.minimum(
                1.0, global_step / tf.math.maximum(1.0, num_warmup_steps))
            learning_rate.assign(target_learning_rate * learning_rate_frac)

        batch_human = next(data_it)

        loss_human = train_step(batch=batch_human)
        epoch_human_losses.append(loss_human.numpy())
        
    average_epoch_human_loss = sum(epoch_human_losses) / len(epoch_human_losses)
    human_losses.append(average_epoch_human_loss)
      
    print(f'\nEpoch {epoch_i}, loss_human: {loss_human.numpy()}, learning_rate {optimizer.learning_rate.numpy()}')

    metrics_human = evaluate_model_PearsonR(extended_model,
                                   dataset=validation_dataset,
                                   max_steps=100)
    val_metric = np.mean([v.numpy() for v in metrics_human.values()])
    val_metrics.append(val_metric)
    print(f'Epoch {epoch_i}, validation PearsonR: {val_metric}')

    if val_metric > best_val_metric:
        best_val_metric = val_metric
        patience_counter = 0
        save_prefix = os.path.join(checkpoint_dir, f'{config}_epoch_{best_val_metric}')
        checkpoint.save(save_prefix)
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"\nEarly stopping triggered after {epoch_i + 1} epochs.")
        print(f"Best validation metric: {best_val_metric}")
        break

plot_losses(human_losses, val_metrics, output_dir, epoch_i + latest_epoch, config)
save_prefix = os.path.join(checkpoint_dir, f'{config}_epoch_{epoch_i + latest_epoch}')
checkpoint.save(save_prefix)