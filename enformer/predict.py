import tensorflow as tf
import enformer
import kipoiseq
from kipoiseq import Interval
import os
import matplotlib.pyplot as plt
import pyfaidx
import seaborn as sns
import numpy as np
import pyBigWig

SEQUENCE_LENGTH = 196608

class FastaStringExtractor:
    
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()

def one_hot_encode(sequence):
  return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

def plot_track(track, interval, title, height=1.5):
    fig, ax = plt.subplots(1, 1, figsize=(20, height))
    ax.fill_between(np.linspace(interval.start, interval.end, num=len(track)), track)
    ax.set_title(title)
    sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    output_file = os.path.join(f'predictions/predictions_target_0.png')
    plt.savefig(output_file)

def plot_tracks(tracks, interval, height=1.5):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()
    output_file = os.path.join(f'predictions/aortic_comparison.png')
    plt.savefig(output_file)

model = enformer.Enformer(channels=1536,
                          num_heads=8,
                          num_transformer_layers=11,
                          pooling_type='max')

extended_model = enformer.ExtendedEnformer(model)
checkpoint = tf.train.Checkpoint(module=extended_model)
#latest = tf.train.latest_checkpoint('checkpoints')
checkpoint.restore('checkpoints/Adam-50-10000_epoch_0.7562907338142395-45')

bw = pyBigWig.open("data/encode_rna/ENCFF281BWX.bigWig")
points = bw.stats("chr11", 47_980_559, 48_177_167, type="mean", nBins=1536)
points = [x if x is not None else 0 for x in points]
original_track = tf.convert_to_tensor(points, dtype=tf.float32)

min_val = tf.reduce_min(original_track)
max_val = tf.reduce_max(original_track)

fasta_extractor = FastaStringExtractor('data/genome.fa')

target_interval = kipoiseq.Interval('chr11', 47_980_559, 48_177_167)

sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)))
predictions = extended_model.predict_on_batch(sequence_one_hot[np.newaxis])[0]

predictions_normalized = (predictions[:, 0] - tf.reduce_min(predictions[:, 0])) / (tf.reduce_max(predictions[:, 0]) - tf.reduce_min(predictions[:, 0]))
predictions_normalized = predictions_normalized * (max_val - min_val) + min_val

plot_track(predictions_normalized, target_interval, 'RNA:aortic smooth muscle cell male adult (predicted)')

tracks = {'RNA:aortic smooth muscle cell male adult (predicted)': predictions_normalized,
          'Experiment': original_track}
plot_tracks(tracks, target_interval)