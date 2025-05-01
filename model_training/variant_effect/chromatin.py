'''Adapted from https://github.com/FunctionLab/ExPecto/blob/master/chromatin.py'''

import argparse
import pyfasta
import numpy as np
import pandas as pd
import pickle

parser = argparse.ArgumentParser(description='Predict variant chromatin effects')
parser.add_argument('inputfile', type=str, help='Input file in vcf format')
parser.add_argument('outfolder', type=str, help='folder to place ref and alt sequences in')
parser.add_argument('--maxshift', action="store",
                    dest="maxshift", type=int, default=800,
                    help='Maximum shift distance for computing nearby effects')
parser.add_argument('--inputsize', action="store", dest="inputsize", type=int,
                    default=2000, help="The input sequence window size for neural network")
parser.add_argument('--batchsize', action="store", dest="batchsize",
                    type=int, default=32, help="Batch size for neural network predictions.")
parser.add_argument('--cuda', action='store_true')

args = parser.parse_args()

# command line arguments
inputfile = args.inputfile
out_folder = args.outfolder
maxshift = args.maxshift
inputsize = args.inputsize
batchSize = args.batchsize
windowsize = inputsize + 100
genome = pyfasta.Fasta('../../data/hg19.fa') # Path to FASTA file


CHRS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
        'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
        'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX','chrY']

def fetch_seqs(chr, pos, ref, alt, shift=0, inputsize=2000):
    """Fetches sequences from the genome.

    Retrieves sequences centered at the given position with the given inputsize.
    Returns both reference and alternative allele sequences . An additional 100bp
    is retrived to accommodate indels.

    Args:
        chr: the chromosome name that must matches one of the names in CHRS.
        pos: chromosome coordinate (1-based).
        ref: the reference allele.
        alt: the alternative allele.
        shift: retrived sequence center position - variant position.
        inputsize: the targeted sequence length (inputsize+100bp is retrived for
                reference allele).

    Returns:
        A string that contains sequence with the reference allele,
        A string that contains sequence with the alternative allele,
        A boolean variable that tells whether the reference allele matches the
        reference genome

        The third variable is returned for diagnostic purpose. Generally it is
        good practice to check whether the proportion of reference allele
        matches is as expected.

    """
    windowsize = inputsize + 100
    mutpos = int(windowsize / 2 - 1 - shift)
    # return string: ref sequence, string: alt sequence, Bool: whether ref allele matches with reference genome
    seq = genome.sequence({'chr': chr, 'start': pos + shift -
                           int(windowsize / 2 - 1), 'stop': pos + shift + int(windowsize / 2)})
    return seq[:mutpos] + ref + seq[(mutpos + len(ref)):], seq[:mutpos] + \
        alt + seq[(mutpos + len(ref)):], \
            seq[mutpos:(mutpos + len(ref))].upper() == ref.upper()

vcf = pd.read_csv(inputfile, sep='\t', header=None, comment='#', compression='gzip')


# standardize
vcf.iloc[:, 0] = 'chr' + vcf.iloc[:, 0].map(str).str.replace('chr', '')
vcf = vcf[vcf.iloc[:, 0].isin(CHRS)]

for shift in [0, ] + list(range(-200, -maxshift - 1, -200)) + list(range(200, maxshift + 1, 200)):
    refseqs = []
    altseqs = []
    ref_matched_bools = []
    for i in range(vcf.shape[0]):
        refseq, altseq, ref_matched_bool = fetch_seqs(
            vcf[0][i], vcf[1][i], vcf[3][i], vcf[4][i], shift=shift, inputsize=inputsize)
        ref_matched_bools.append(ref_matched_bool)

        if ref_matched_bool:
            refseqs.append(refseq)
            altseqs.append(altseq)

    if shift == 0:
        # only need to be checked once
        print("Number of variants with reference allele matched with reference genome:")
        print(np.sum(ref_matched_bools))
        print("Number of input variants:")
        print(len(ref_matched_bools))

    with open(f'{out_folder}/refseqs_human' + '.shift_' + str(shift), 'wb') as fp:
        pickle.dump(refseqs, fp)

    with open(f'{out_folder}/altseqs_human' + '.shift_' + str(shift), 'wb') as fp:
        pickle.dump(altseqs, fp)


