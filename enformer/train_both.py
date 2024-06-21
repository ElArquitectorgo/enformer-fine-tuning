import tensorflow as tf
import sonnet as snt
from tqdm import tqdm
import numpy as np
import os
import enformer
from dataset import get_dataset
from train_utils import evaluate_extended_model, plot_both_losses

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def create_step_function(model, optimizer):

  @tf.function
  def train_step(batch, head, optimizer_clip_norm_global=0.2):
    with tf.GradientTape() as tape:
      outputs = model(batch['sequence'], is_training=True)[head]
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
extended_model = enformer.ExtendedEnformerMouse(model)

train_step = create_step_function(extended_model, optimizer)
config = 'Adam-400'
steps_per_epoch = 400
num_epochs = 200

human_dataset = get_dataset('human', 'train').batch(1).repeat()
mouse_dataset = get_dataset('mouse', 'train').batch(1).repeat()
human_mouse_dataset = tf.data.Dataset.zip((human_dataset, mouse_dataset)).prefetch(2)
human_valid_dataset = get_dataset('human', 'valid').batch(1).prefetch(2)
mouse_valid_dataset = get_dataset('mouse', 'valid').batch(1).prefetch(2)

data_it = iter(human_mouse_dataset)
global_step = 0

output_dir = 'learning_curves_both'
os.makedirs(output_dir, exist_ok=True)

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint = tf.train.Checkpoint(module=extended_model)
human_losses = []
mouse_losses = []
human_val_metrics = []
mouse_val_metrics = []
patience = 20
best_human_val_metric = -np.inf
best_mouse_val_metric = -np.inf
patience_counter = 0

for epoch_i in range(num_epochs):
    epoch_human_losses = []
    epoch_mouse_losses = []
    for i in tqdm(range(steps_per_epoch)):
        global_step += 1

        if global_step > 1:
            learning_rate_frac = tf.math.minimum(
                1.0, global_step / tf.math.maximum(1.0, num_warmup_steps))
            learning_rate.assign(target_learning_rate * learning_rate_frac)

        batch_human, batch_mouse = next(data_it)

        loss_human = train_step(batch=batch_human, head='human')
        loss_mouse = train_step(batch=batch_mouse, head='mouse')

        epoch_human_losses.append(loss_human.numpy())
        epoch_mouse_losses.append(loss_mouse.numpy())
        
    average_epoch_human_loss = sum(epoch_human_losses) / len(epoch_human_losses)
    human_losses.append(average_epoch_human_loss)

    average_epoch_mouse_loss = sum(epoch_mouse_losses) / len(epoch_mouse_losses)
    mouse_losses.append(average_epoch_mouse_loss)
      
    print(f'\nEpoch {epoch_i}, loss_human: {average_epoch_human_loss}, learning_rate {optimizer.learning_rate.numpy()}')
    print(f'Epoch {epoch_i}, loss_mouse: {average_epoch_mouse_loss}, learning_rate {optimizer.learning_rate.numpy()}')

    metrics_human = evaluate_extended_model(extended_model,
                                   dataset=human_valid_dataset,
                                   corr='PearsonR',
                                   head='human',
                                   max_steps=100)
    human_val_metric = np.mean([v.numpy() for v in metrics_human.values()])
    human_val_metrics.append(human_val_metric)
    print(f'Human validation PearsonR: {human_val_metric}')

    if human_val_metric > best_human_val_metric:
        best_human_val_metric = human_val_metric
        patience_counter = 0
        save_prefix = os.path.join(checkpoint_dir, f'extended_checkpoints/{config}_human_{best_human_val_metric}')
        checkpoint.save(save_prefix)
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"\nEarly stopping triggered after {epoch_i + 1} epochs.")
        print(f"Best validation metric: {best_human_val_metric}")
        break

    metrics_mouse = evaluate_extended_model(extended_model,
                                   dataset=mouse_valid_dataset,
                                   corr='PearsonR',
                                   head='mouse',
                                   max_steps=100)
    mouse_val_metric = np.mean([v.numpy() for v in metrics_mouse.values()])
    mouse_val_metrics.append(mouse_val_metric)
    print(f'Mouse validation PearsonR: {mouse_val_metric}')

plot_both_losses(human_losses, human_val_metrics, mouse_losses, mouse_val_metrics, output_dir, epoch_i, config)