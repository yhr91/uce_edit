import scanpy as sc
import argparse
from evaluate import AnndataProcessor
from accelerate import Accelerator

default_arguments = {
    # Anndata Processing Arguments
    'adata_path': '/dfs/user/yhr/Arc/uce_edit/embryonic_stem_cells.h5ad',
    'dir': "./",
    'species': "human",
    'filter': True,
    'skip': True,

    # Model Arguments
    'model_loc': None,
    'batch_size': 25,
    'pad_length': 1536,
    'pad_token_idx': 0,
    'chrom_token_left_idx': 1,
    'chrom_token_right_idx': 2,
    'cls_token_idx': 3,
    'CHROM_TOKEN_OFFSET': 143574,
    'sample_size': 1024,
    'CXG': True,
    'nlayers': 4,
    'output_dim': 1280,
    'd_hid': 5120,
    'token_dim': 5120,
    'multi_gpu': False,

    # Misc Arguments
    'spec_chrom_csv_path': "./model_files/species_chrom.csv",
    'token_file': "./model_files/all_tokens.torch",
    'protein_embeddings_dir': "./model_files/protein_embeddings/",
    'offset_pkl_path': "./model_files/species_offsets.pkl"
}
args = argparse.Namespace(**default_arguments)

accelerator = Accelerator(project_dir=args.dir)
processor = AnndataProcessor(args, accelerator)
processor.preprocess_anndata()
processor.generate_idxs()

processor.run_evaluation(perturb_gene='FOS', perturb_level=5)

