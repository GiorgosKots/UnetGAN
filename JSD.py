import torch
import numpy as np
from itertools import product
from collections import Counter
from scipy.spatial.distance import jensenshannon

# Set device based on availability of GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Convert one-hot or Gumbel-softmax output to DNA string
def tensor_to_dna(seq_tensor):
    idx_to_base = ['P', 'A', 'T', 'G', 'C']
    indices = torch.argmax(seq_tensor, dim=-1)
    return ''.join([idx_to_base[i] for i in indices.tolist() if idx_to_base[i] != 'P'])

# 2. Get k-mer distribution
def get_kmer_distribution(seqs, k=6):
    all_kmers = [''.join(p) for p in product('ACGT', repeat=k)]
    kmer_counts = Counter()
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmer_counts[kmer] += 1
    total = sum(kmer_counts.values())
    freq = np.array([kmer_counts[kmer] for kmer in all_kmers], dtype=np.float32)
    return freq / (total + 1e-8)

# 3. Evaluate per epoch
def evaluate_kmer_jsd(real_batch, gen_batch, k=6):
    real_seqs = [tensor_to_dna(seq) for seq in real_batch]  # shape: (batch_size, seq_len, 4)
    gen_seqs  = [tensor_to_dna(seq) for seq in gen_batch]

    real_dist = get_kmer_distribution(real_seqs, k=k)
    gen_dist  = get_kmer_distribution(gen_seqs, k=k)

    jsd = jensenshannon(real_dist, gen_dist)
    return jsd

def jsd(generator, dataloader, num_batches=5, k=6):
    generator.eval()  # Set to evaluation mode
    total_jsd = 0.0

    with torch.no_grad():
        for i in range(num_batches):
            # Get real batch
            try:
                real_batch = next(iter(dataloader)).to(device)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                real_batch = next(dataloader_iter).to(device)

            # Generate fake batch
            noise = torch.randn(real_batch.size(0), 128).to(device)
            fake_batch = generator(noise)

            # Calculate JSD for this batch
            batch_jsd = evaluate_kmer_jsd(real_batch, fake_batch, k=k)
            total_jsd += batch_jsd

    # Calculate average JSD
    avg_jsd = total_jsd / num_batches
    return avg_jsd

# Example usage:
# from JSD import jsd
# avg_jsd = jsd(generator, dataloader)
