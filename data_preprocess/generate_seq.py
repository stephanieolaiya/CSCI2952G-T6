import pyfasta
import pandas as pd


genome = pyfasta.Fasta('./data/hg19.fa') # Path to FASTA file
tss_offset = 1000  # Extend TSS by this many bp upstream and downstream

def fetch_gene_sequence(chromosome, tss, strand):
    '''
    Extract the string sequence for a given gene which is the a sequence of length 
    tss_offset * 2 centered around the TSS site of the gene. 
    '''
    start_pos = max(1, tss - tss_offset)  # Ensure start is not negative
    end_pos = tss + tss_offset
    seq = genome.sequence({'chr': chromosome, 'start': start_pos, 'stop': end_pos, 'strand': strand})
    return seq

def create_dataset():
    # gene annotation file corresponding to tissue expression data
    geneanno = pd.read_csv('./data/geneanno.csv') 
    geneanno_exp = pd.read_csv('./data/geneanno.exp.csv')
    geneanno_merged =  pd.merge(geneanno, 
                                geneanno_exp, 
                                left_index=True, 
                                right_index=True, 
                                how='outer')
    geneanno_merged = geneanno_merged[(geneanno_merged['seqnames'] != 'chrX') 
                                      & (geneanno_merged['seqnames'] != 'chrY')]
    geneanno_merged['seq'] = geneanno_merged.apply(lambda x: 
                                                   fetch_gene_sequence(x.seqnames, 
                                                                       x.TSS, 
                                                                       x.strand),
                                                                       axis=1)
    geneanno_merged.to_csv('./data/sequence_exp.csv')

def main():
    create_dataset()

if __name__ == "__main__":
    main()
