#!/bin/bash

# GWAS catalog
wget ftp://ftp.ebi.ac.uk/pub/databases/gwas/releases/2018/12/27/gwas-catalog-associations.tsv

# Base URL
BASE_URL="ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"

# Chromosomes to download (1-22)
CHROMS=({1..22})

# Loop through chromosomes and download VCF files
for CHR in "${CHROMS[@]}"; do
    FILE="ALL.chr${CHR}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    echo "Downloading $FILE..."
    wget "${BASE_URL}/${FILE}"
    wget "${BASE_URL}/${FILE}.tbi"
done