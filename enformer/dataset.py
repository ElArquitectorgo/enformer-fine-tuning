import tensorflow as tf
import os
import json
import functools

def get_dataset(organism, subset, num_threads=8):
  metadata = get_metadata(organism)
  dataset = tf.data.TFRecordDataset(tfrecord_files(organism, subset),
                                    compression_type='ZLIB',
                                    num_parallel_reads=num_threads)
  dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                        num_parallel_calls=num_threads)
  return dataset


def get_metadata(organism):
  with open(f'data/{organism}/statistics.json', 'r') as f:
        return json.load(f)


def tfrecord_files(organism, subset):
    tfr_dir = os.getcwd() + f'/data/{organism}/tfrecords'
    tfr_files = [os.path.join(tfr_dir, file) for file in os.listdir(tfr_dir) if file.startswith(f'{subset}-') and file.endswith('.tfr')]
    return sorted(tfr_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))


def deserialize(serialized_example, metadata):
  """Deserialize bytes stored in TFRecordFile."""
  feature_map = {
      'sequence': tf.io.FixedLenFeature([], tf.string),
      'target': tf.io.FixedLenFeature([], tf.string),
  }
  example = tf.io.parse_example(serialized_example, feature_map)
  sequence = tf.io.decode_raw(example['sequence'], tf.bool)
  sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
  sequence = tf.cast(sequence, tf.float32)

  target = tf.io.decode_raw(example['target'], tf.float16)
  target = tf.reshape(target,
                      (metadata['target_length'], metadata['num_targets']))
  target = tf.cast(target, tf.float32)

  return {'sequence': sequence,
          'target': target}