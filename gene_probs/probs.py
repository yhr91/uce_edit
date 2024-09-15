import torch
import torch.nn as nn
from tqdm import tqdm
from uce_model import TransformerModel

class UCEGenePredictor:
    """
    Class to predict binary gene expression from UCE embeddings

    Usage:
    gene_predictor = UCEGenePredictor(device='cuda:0')
    cell_embeds = sc.read_h5ad('path_to_uce_embeds')
    gene_logprobs = gene_predictor.compute_gene_prob_group(genes, cell_embeds)

    """

    def __init__(self,
                 model_loc='/dfs/project/cross-species/yanay/code/state_dicts/model_used_in_paper_33l_8ep_1024t_1280.torch',
                 gene_idx_file="/dfs/user/yhr/snap/UCE/new_tabula_sapiens_ep_8_sn_251656_nlayers_4_sample_size_1024_pe_idx.torch",
                 token_file='/dfs/user/yhr/snap/UCE/model_files/all_tokens.torch',
                 protein_embeds_file='/dfs/user/yhr/snap/UCE/model_files/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
                 token_dim=5120, emsize=1280, d_hid=5120, nlayers=33, nhead=20,
                 dropout=0.05, output_dim=1280,
                 device='cuda:0'):

        self.device = device
        self.model_loc = model_loc
        self.gene_idx = torch.load(gene_idx_file)
        self.gene_idx = list(self.gene_idx.values())[0]
        self.token_file = token_file
        self.protein_embeds = torch.load(protein_embeds_file)
        self.model = TransformerModel(token_dim=token_dim,
                                      d_model=emsize, nhead=nhead,
                                      d_hid=d_hid, nlayers=nlayers,
                                      dropout=dropout,
                                      output_dim=output_dim)
        self.model = self.model.to(self.device)

        self.all_pe = self.get_ESM2_embeddings()
        self.all_pe.requires_grad = False
        self.model.pe_embedding = nn.Embedding.from_pretrained(self.all_pe)
        self.model.load_state_dict(
            torch.load(self.model_loc, map_location="cpu"), strict=True)
        self.model = self.model.eval()

    def get_ESM2_embeddings(self):
        all_pe = torch.load(self.token_file)
        if all_pe.shape[0] == 143574:
            torch.manual_seed(23)
            CHROM_TENSORS = torch.normal(mean=0, std=1,
                                         size=(1895, self.token_dim))
            all_pe = torch.vstack((all_pe, CHROM_TENSORS))
            all_pe.requires_grad = False

        return all_pe

    def get_reduced_embeds(self, genes):
        return self.model.gene_embedding_layer(
            torch.stack([self.protein_embeds[x] for x in genes]).to(self.device))

    def get_MLP_input(self, cell_embed, task_embeds):
        A = task_embeds
        B = torch.Tensor(cell_embed).unsqueeze(1).repeat(1,
                                                         task_embeds.shape[0]).T
        mlp_input = torch.cat([A, B], 1)
        return mlp_input

    def compute_gene_prob(self, genenames, cell_embed):
        task_embeds = self.get_reduced_embeds(genenames)
        mlp_input = self.get_MLP_input(cell_embed, task_embeds)
        mlp_input = mlp_input.to(self.device)
        return self.model.binary_decoder(mlp_input).detach().cpu()

    def compute_gene_prob_group(self, cell_embeds, genenames):
        all_logprobs = []
        for cell_embed in tqdm(cell_embeds, total=len(cell_embeds)):
            cell_embed = torch.Tensor(cell_embed).to(self.device)
            all_logprobs.append(self.compute_gene_prob(genenames, cell_embed))
        return all_logprobs