import encode
import numpy as np
from Bio import SeqIO
from multiprocessing import Pool
from functools import partial
import torch
from torch.utils.data import Dataset
import re


class BioDataset(Dataset):
    def __init__(self, features, labels):
        """
        :param features: (n_samples, seq_len, num_features)
        :param labels: (n_samples,)
        """
        self.features = torch.tensor(features, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        """
        Returns a sample at the given index idx.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns a sample at the given index idx.
        :param idx: index of the sample
        """
        return self.features[idx], self.labels[idx]


class Biodata:
    def __init__(self, pos_fasta_files, neg_fasta_files, k=6):
        self.feature = None
        self.dna_seq = {}
        self.label = []
        for pos_fasta_file in pos_fasta_files:
            for seq_record in SeqIO.parse(pos_fasta_file, "fasta"):
                sequence = str(seq_record.seq).upper()
                sequence = re.sub(r'[^ATCG]', 'C', sequence)
                if sequence:
                    if seq_record.description not in self.dna_seq:
                        self.dna_seq[seq_record.description] = sequence
                        self.label.append(1)
        for neg_fasta_file in neg_fasta_files:
            for seq_record in SeqIO.parse(neg_fasta_file, "fasta"):
                sequence = str(seq_record.seq).upper()
                sequence = re.sub(r'[^ATCG]', 'T', sequence)
                if sequence:
                    if seq_record.description not in self.dna_seq:
                        self.dna_seq[seq_record.description] = sequence
                        self.label.append(0)
        self.k = k
        self.label = np.array(self.label)

    def encode_seq(self, thread):
        print("encoding sequences...")
        seq_list = list(self.dna_seq.values())

        # DNA2VEC
        pool = Pool(thread)
        partial_encode_seq = partial(encode.matrix_encoding, k=self.k)
        self.feature = np.array(pool.map(partial_encode_seq, seq_list))

        # One-hot
        # self.feature = np.array(pool.map(encode.one_hot_matrix(seq_list)))

        pool.close()
        pool.join()

        dataset = BioDataset(self.feature, self.label)
        return dataset
