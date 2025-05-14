# CSCI2952G-T6: Deep Learning In Genomics Final Project

GeneExpress is a framework that finetunes the Nucleotide Transformer 2.5B Multi-Species (NT) model—pretrained on genomic sequences from over 850 species—for gene expression prediction (GEP). Therefore the model outputs predicted gene expression levels across 218 human tissues, taking in 6kb DNA sequences as input. 

This repository contains code for:
- Model training and evaluation
- Variant effect prediction (VEP)
- Cross-Species gene expression prediction

These results highlight the potential of multi-species foundation models for scalable, tissue-specific, and cross-species GEP directly from DNA sequences.

- `/data_preprocess`: sequence generation from hg19 FASTA file
- `/model_training`: model training and evaluation
- `/model_training/saved_downstream`: saved model weights (downstream_weights.pt is most current version)
- `/model_training/variant_effect`: Disease specific cvariant effect prediction
- `/cross_species_eval`: cross species gene expression prediction
- `/download_resources.sh`: download hg19.fa and other needed resources for sequence generation
- `/download_resources_mouse.sh`: download mouse FASTA files and other needed resources for sequence generation
- `vep/vcf_files`: download resources for vcf file creation and code to filter for specific diseases.


