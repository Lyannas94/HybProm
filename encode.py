import numpy as np

embeddings = np.load(r'embedding_matrix.npy')
# replace the path of embedding_matrix.npy

def kmer_to_number(kmer):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    number = 0
    for char in kmer:
        number = number * 4 + mapping[char]
    return number + 1


def generate_kmers(dna_sequence, k):
    kmers = []
    n = len(dna_sequence)
    for i in range(n - k + 1):
        kmer = dna_sequence[i:i + k]
        kmer_number = kmer_to_number(kmer)
        kmers.append(kmer_number)
    return kmers


def map_kmers_to_embedding(kmers_numbers, embedding):
    mapped_embeddings = np.array([embedding[number - 1] for number in kmers_numbers])
    return mapped_embeddings


def dna_to_one_hot(dna_sequence):
    base_to_one_hot = {
        'A': np.array([1, 0, 0, 0]),
        'T': np.array([0, 1, 0, 0]),
        'C': np.array([0, 0, 1, 0]),
        'G': np.array([0, 0, 0, 1])
    }
    one_hot_sequence = np.zeros((len(dna_sequence), 4))

    for i, base in enumerate(dna_sequence):
        one_hot_sequence[i] = base_to_one_hot[base]

    return one_hot_sequence


def matrix_encoding(seq, k):
    seq = seq.upper()
    kmers_numbers = generate_kmers(seq, k)
    matrix = map_kmers_to_embedding(kmers_numbers, embeddings)
    return matrix


def one_hot_matrix(seq):
    seq = seq.upper()
    matrix = dna_to_one_hot(seq)
    return matrix

# test_Seq = 'ATGCGATCGTTTTATTAT'

# dna2vec = matrix_encoding(test_Seq, 6)
# one_hot = dna_to_one_hot(test_Seq)
