#!/bin/sh
wget ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M21/GRCm38.primary_assembly.genome.fa.gz
gunzip GRCm38.primary_assembly.genome.fa.gz
# Clean up FASTA headers to fit generate_seq_mouse.py
sed 's/^\(>[^ ]*\).*/\1/' GRCm38.primary_assembly.genome.fa > GRCm38.cleaned.fa