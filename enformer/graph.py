# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tensorflow implementation of Enformer model.

"Effective gene expression prediction from sequence by integrating long-range
interactions"

Žiga Avsec1, Vikram Agarwal2,4, Daniel Visentin1,4, Joseph R. Ledsam1,3,
Agnieszka Grabska-Barwinska1, Kyle R. Taylor1, Yannis Assael1, John Jumper1,
Pushmeet Kohli1, David R. Kelley2*

1 DeepMind, London, UK
2 Calico Life Sciences, South San Francisco, CA, USA
3 Google, Tokyo, Japan
4 These authors contributed equally.
* correspondence: avsec@google.com, pushmeet@google.com, drk@calicolabs.com
"""

import numpy as np
import tensorflow as tf
import tensorflow as tf
import kipoiseq
from kipoiseq import Interval
import os
import matplotlib.pyplot as plt
import pyfaidx
import seaborn as sns
import numpy as np
import enformer

SEQUENCE_LENGTH = 196_608

def plot_tracks(tracks, interval, height=1.5):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()
    output_file = os.path.join(f'predictions/graph.png')
    plt.savefig(output_file)

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

def one_hot_encode2(sequence):
  return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

model = enformer.Enformer(channels=1536,
                          num_heads=8,
                          num_transformer_layers=11,
                          pooling_type='max')

extended_model = enformer.ExtendedEnformer(model)
checkpoint = tf.train.Checkpoint(module=extended_model)
#latest = tf.train.latest_checkpoint('checkpoints')
checkpoint.restore('checkpoints/Adam-50-10000_epoch_0.7562907338142395-45')
fasta_extractor = FastaStringExtractor('data/genome.fa')

target_interval = kipoiseq.Interval('chr11', 48_070_696, 48_267_304)  # centro -> 48,169,000

sequence_one_hot = one_hot_encode2(fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)))
predictions = extended_model.predict_on_batch(sequence_one_hot[np.newaxis])[0]

target_mask = np.zeros_like(predictions)
for idx in [764, 765, 766, 767, 768, 769, 770, 771, 772]:
  target_mask[idx, 0] = 1

contribution_scores = extended_model.contribution_input_grad(sequence_one_hot.astype(np.float32), target_mask).numpy()
pooled_contribution_scores = tf.nn.avg_pool1d(np.abs(contribution_scores)[np.newaxis, :, np.newaxis], 128, 128, 'VALID')[0, :, 0].numpy()
tracks = {'RNA:aortic smooth muscle cell male predictions': predictions[:, 0],
          'Model gradient*input': np.minimum(pooled_contribution_scores, 0.03)}
plot_tracks(tracks, target_interval)