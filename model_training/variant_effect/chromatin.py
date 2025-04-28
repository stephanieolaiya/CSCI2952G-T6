'''Adapted from https://github.com/FunctionLab/ExPecto/blob/master/chromatin.py'''

import argparse
import pyfasta
import torch
import numpy as np
import pandas as pd
import h5py
from transformers import AutoTokenizer, AutoModel
from model.model import ExpressionPredictor

parser = argparse.ArgumentParser(description='Predict variant chromatin effects')
parser.add_argument('inputfile', type=str, help='Input file in vcf format')
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
maxshift = args.maxshift
inputsize = args.inputsize
batchSize = args.batchsize
windowsize = inputsize + 100

genome = pyfasta.Fasta('../../data/hg19.fa') # Path to FASTA file

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
model_name = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
max_length = tokenizer.model_max_length

# Load downstream model
output_dim = 218 # number of uhman tissues
model = ExpressionPredictor(base_model, output_dim).to(device)
model.mlp.load_state_dict(torch.load("./saved_downstream/mlp_weights.pt"))
model.eval()


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
        refseqs.append(refseq)
        altseqs.append(altseq)
        ref_matched_bools.append(ref_matched_bool)


    all_diff =  []
    # get gene expression value differences
    for i in range(len(refseqs)):
        ref_tokenized = tokenizer(refseqs[i], return_tensors="pt", padding="max_length", max_length=self.max_len, truncation=True)
        ref_input_ids = ref_tokenized["input_ids"].squeeze(0)  # (seq_len,)
        ref_attention_mask = (ref_input_ids != tokenizer.pad_token_id).long().to(device)
        alt_tokenized = tokenizer(altseqs[i], return_tensors="pt", padding="max_length", max_length=self.max_len, truncation=True)
        alt_input_ids = alt_tokenized["input_ids"].squeeze(0)  # (seq_len,)
        alt_attention_mask = (alt_input_ids != tokenizer.pad_token_id).long().to(device)
        # get model predictons for both
        ref_preds = model(ref_input_ids, ref_attention_mask).float()
        alt_preds = model(alt_input_ids, alt_attention_mask).float()
        diff = alt_preds - ref_preds   
        all_diff.append(np.array(diff))   


    f = h5py.File(inputfile + '.shift_' + str(shift) + '.diff.h5', 'w')
    f.create_dataset('pred', data=diff)
    f.close()


