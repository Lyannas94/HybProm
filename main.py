import torch
import model
from multiprocessing import Process
import Biodata
import train
# from other_models import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def foo(i):
    print(" This is Process ", i)


def main():
    for i in range(5):
        p = Process(target=foo, args=(i,))
        p.start()


if __name__ == '__main__':
    # # ------------------------------Train-----------------------------
    # pos_train_files = ['datasets/D. melanogaster_Non_TATA/Pos_train.fasta']
    # neg_train_files = ['datasets/D. melanogaster_Non_TATA/Neg_train.fasta']
    #
    # # pos_train_files = ['datasets/D. melanogaster_TATA/Pos_train.fasta']
    # # neg_train_files = ['datasets/D. melanogaster_TATA/Neg_train.fasta']
    #
    # # pos_train_files = ['datasets/Escherichia coli K-12/sigma24.fasta',
    # #                    'datasets/Escherichia coli K-12/sigma28.fasta',
    # #                    'datasets/Escherichia coli K-12/sigma32.fasta',
    # #                    'datasets/Escherichia coli K-12/sigma38.fasta',
    # #                    'datasets/Escherichia coli K-12/sigma54.fasta',
    # #                    'datasets/Escherichia coli K-12/sigma70.fasta']
    # # neg_train_files = ['datasets/Escherichia coli K-12/Neg_train.fasta']
    # #
    # # pos_train_files = ['datasets/H.spanies_Non_TATA/Pos_train.fasta']
    # # neg_train_files = ['datasets/H.spanies_Non_TATA/Neg_train.fasta']
    # #
    # # pos_train_files = ['datasets/H.spanies_TATA/Pos_train.fasta']
    # # neg_train_files = ['datasets/H.spanies_TATA/Neg_train.fasta']
    #
    # # pos_train_files = ['datasets/M.musculus_Non_TATA/Pos_train.fasta']
    # # neg_train_files = ['datasets/M.musculus_Non_TATA/Neg_train.fasta']
    #
    # # pos_train_files = ['datasets/M.musculus_TATA/Pos_train.fasta']
    # # neg_train_files = ['datasets/M.musculus_TATA/Neg_train.fasta']
    #
    # # pos_train_files = ['datasets/Plants/Pos_train.fasta']
    # # neg_train_files = ['datasets/Plants/Neg_train.fasta']
    #
    # data_train = Biodata.Biodata(pos_train_files, neg_train_files, k=6)
    # train_set = data_train.encode_seq(thread=20)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_model = model.CNNBiLSTMAttention().to(device)
    # train.train_model(train_set, train_model)

    # ----------------------------Test----------------------------------
    # pos_test_files = ['datasets/D. melanogaster_Non_TATA/Pos_test.fasta']
    # neg_test_files = ['datasets/D. melanogaster_Non_TATA/Neg_test.fasta']

    # pos_test_files = ['datasets/D. melanogaster_TATA/Pos_test.fasta']
    # neg_test_files = ['datasets/D. melanogaster_TATA/Neg_test.fasta']

    pos_test_files = ['datasets/Escherichia coli K-12/Pos_test.fasta']
    neg_test_files = ['datasets/Escherichia coli K-12/Neg_test.fasta']

    # pos_test_files = ['datasets/H.spanies_Non_TATA/Pos_test.fasta']
    # neg_test_files = ['datasets/H.spanies_Non_TATA/Neg_test.fasta']
    #
    # pos_test_files = ['datasets/H.spanies_TATA/Pos_test.fasta']
    # neg_test_files = ['datasets/H.spanies_TATA/Neg_test.fasta']
    #
    # pos_test_files = ['datasets/M.musculus_Non_TATA/Pos_test.fasta']
    # neg_test_files = ['datasets/M.musculus_Non_TATA/Neg_test.fasta']
    #
    # pos_test_files = ['datasets/M.musculus_TATA/Pos_test.fasta']
    # neg_test_files = ['datasets/M.musculus_TATA/Neg_test.fasta']
    #
    # pos_test_files = ['datasets/Plants/Pos_test.fasta']
    # neg_test_files = ['datasets/Plants/Neg_test.fasta']

    data_test = Biodata.Biodata(pos_test_files, neg_test_files, k=6)
    test_set = data_test.encode_seq(thread=20)
    train.test_model(test_set)
