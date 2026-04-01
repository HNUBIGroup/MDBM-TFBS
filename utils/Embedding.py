import numpy as np

def one_hot_kmer(seq, k):

    base_map = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]}

    seq_len = len(seq)
    num_kmers = seq_len - k + 1

    code = np.zeros(shape=(seq_len, 4 * k))

    for i in range(num_kmers):
        kmer = seq[i:i+k]
        code[i] = np.concatenate([base_map[base] for base in kmer])

    return code


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers