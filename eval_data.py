"""
Dataloaders

"""

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
import pickle
import torch.utils.data as data


class MultiDatasetSentences(data.Dataset):
    def __init__(self, sorted_dataset_names, shapes_dict, args, 
                 dataset_to_protein_embeddings_path= "/lfs/local/0/yanay/reduced_datasets_to_pe_chrom_5120_new.torch",
                 datasets_to_chroms_path="/lfs/local/0/yanay/dataset_to_chroms_new.pkl",
                 datasets_to_starts_path="/lfs/local/0/yanay/dataset_to_starts_new.pkl",
                 npzs_dir="/lfs/local/0/yanay/uce_proc/",
                 perturb_gene=None, 
                 perturb_level=5) -> None:
        super(MultiDatasetSentences, self).__init__()
        # self.xs = {}
        self.num_cells = {}
        self.num_genes = {}
        self.shapes_dict = shapes_dict
        self.args = args
        self.perturb_gene = perturb_gene
        self.perturb_level = perturb_level

        self.total_num_cells = 0
        for name in sorted_dataset_names:
            num_cells, num_genes = self.shapes_dict[name]
            # self.xs[name] = X
            self.num_cells[name] = num_cells
            self.num_genes[name] = num_genes

            self.total_num_cells += num_cells

        self.datasets = sorted_dataset_names

        # TODO: preferably not hard-coded here
        self.dataset_to_protein_embeddings = torch.load(dataset_to_protein_embeddings_path)
        with open(datasets_to_chroms_path, "rb") as f:
            self.dataset_to_chroms = pickle.load(f)
        with open(datasets_to_starts_path, "rb") as f:
            self.dataset_to_starts = pickle.load(f)
        
        self.npzs_dir = npzs_dir

    def __getitem__(self, idx):
        if isinstance(idx, int):
            for dataset in sorted(self.datasets):
                if idx < self.num_cells[dataset]:
                    #cts = np.memmap(f"/lfs/local/0/yanay/cxg_npzs/" + f"{dataset}_counts.npz",
                    #        dtype='int64', mode='r', shape=self.shapes_dict[dataset])
                    cts = np.memmap(self.npzs_dir + f"{dataset}_counts.npz", dtype='int64', mode='r', shape=self.shapes_dict[dataset])
                    counts = cts[idx]
                    counts = torch.tensor(counts).unsqueeze(0)
                    weights = torch.log1p(counts)
                    weights = (weights / torch.sum(weights))
                    if self.perturb_gene is None:
                        batch_sentences, mask, seq_len, cell_sentences = \
                            sample_cell_sentences(counts, weights, dataset, self.args,
                                dataset_to_protein_embeddings= self.dataset_to_protein_embeddings,
                                dataset_to_chroms=self.dataset_to_chroms,
                                dataset_to_starts=self.dataset_to_starts)
                        perturb_flag = None
                    else:
                        batch_sentences, mask, seq_len, cell_sentences, \
                            perturb_flag = \
                            sample_perturbed_cell_sentences(counts, weights, dataset, self.args,
                                dataset_to_protein_embeddings= self.dataset_to_protein_embeddings,
                                dataset_to_chroms=self.dataset_to_chroms,
                                dataset_to_starts=self.dataset_to_starts, 
                                perturb_gene=self.perturb_gene,
                                perturb_level=self.perturb_level)
                    return batch_sentences, mask, idx, seq_len, \
                        cell_sentences, perturb_flag
                else:
                    idx -= self.num_cells[dataset]
            raise IndexError
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return self.total_num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes


class MultiDatasetSentenceCollator(object):
    def __init__(self, args):
        self.pad_length = args.pad_length


    def __call__(self, batch):
        batch_size = len(batch)
        ## If perturbation setting
        if len(batch[0][0]) == 2:
            batch_sentences = torch.zeros((batch_size*2, self.pad_length))
            mask = torch.zeros((batch_size*2, self.pad_length))
            cell_sentences = torch.zeros((batch_size*2, self.pad_length))
            perturb_flag = torch.zeros(batch_size*2)
            idxs = torch.zeros(batch_size*2)

        ## if not perturbation setting
        else:
            batch_sentences = torch.zeros((batch_size, self.pad_length))
            mask = torch.zeros((batch_size, self.pad_length))
            cell_sentences = torch.zeros((batch_size, self.pad_length))
            idxs = torch.zeros(batch_size)
            perturb_flag = torch.zeros(batch_size)

        i = 0
        max_len = 0

        for bs, msk, idx, seq_len, cs, pflag in batch:
            if len(batch[0][0]) == 2:
                batch_sentences[i:i+2, :] = torch.stack(bs).squeeze()
                cell_sentences[i:i+2, :] = torch.stack(cs).squeeze()
                max_len = max(max_len, seq_len)
                mask[i:i+2, :] = torch.stack(msk).squeeze()
                idxs[i:i+2] = torch.Tensor([idx, idx])
                perturb_flag[i:i+2] = torch.Tensor(pflag).squeeze()

                i += 2
            else:
                batch_sentences[i, :] = bs
                cell_sentences[i, :] = cs
                max_len = max(max_len, seq_len)
                mask[i, :] = msk
                idxs[i] = idx

                i += 1

        return batch_sentences[:, :max_len] , mask[:, :max_len], idxs, \
            cell_sentences, perturb_flag




def sample_cell_sentences(counts, batch_weights, dataset, args,
                          dataset_to_protein_embeddings,
                          dataset_to_chroms,
                          dataset_to_starts):

    dataset_idxs = dataset_to_protein_embeddings[dataset] # get the dataset specific protein embedding idxs
    cell_sentences = torch.zeros((counts.shape[0], args.pad_length)) # init the cell representation as 0s
    mask = torch.zeros((counts.shape[0], args.pad_length)) # start of masking the whole sequence
    chroms = dataset_to_chroms[dataset] # get the dataset specific chroms for each gene
    starts = dataset_to_starts[dataset] # get the dataset specific genomic start locations for each gene

    longest_seq_len = 0 # we need to keep track of this so we can subset the batch at the end

    for c, cell in enumerate(counts):
        weights = batch_weights[c].numpy()
        weights = weights / sum(weights)  # RE NORM after mask
        
        # randomly choose the genes that will make up the sample, weighted by expression, with replacement
        choice_idx = np.random.choice(np.arange(len(weights)),
                                      size=args.sample_size, p=weights,
                                      replace=True)
        choosen_chrom = chroms[choice_idx] # get the sampled genes chromosomes
        # order the genes by chromosome
        chrom_sort = np.argsort(choosen_chrom)  
        choice_idx = choice_idx[chrom_sort]

        # sort the genes by start
        new_chrom = chroms[choice_idx]
        choosen_starts = starts[choice_idx]

        ordered_choice_idx = np.full((args.pad_length),
                                     args.cls_token_idx)  # start with cls
        # i= 0 first token is CLS
        i = 1  # continue on to the rest of the sequence with left bracket being assumed.
        # Shuffle the chroms now, there's no natural order to chromosomes
        uq_chroms = np.unique(new_chrom)
        np.random.shuffle(uq_chroms) # shuffle
        
        # This loop is actually just over one cell
        for chrom in uq_chroms:
            # Open Chrom token
            ordered_choice_idx[i] = int(chrom) + args.CHROM_TOKEN_OFFSET # token of this chromosome # i = 1 next token is a chrom open
            i += 1
            # now sort the genes by start order within the chroms
            loc = np.where(new_chrom == chrom)[0]
            sort_by_start = np.argsort(
                choosen_starts[loc])  # start locations for this chromsome

            to_add = choice_idx[loc[sort_by_start]]
            ordered_choice_idx[i:(i + len(to_add))] = dataset_idxs[to_add]
            i += len(to_add)
            ordered_choice_idx[i] = args.chrom_token_right_idx # add the chrom sep again
            i += 1  # add the closing token again

        longest_seq_len = max(longest_seq_len, i)
        remainder_len = (args.pad_length - i)

        cell_mask = torch.concat((torch.ones(i),
                                  # pay attention to all of these tokens, ignore the rest!
                                  torch.zeros(remainder_len)))

        mask[c, :] = cell_mask

        ordered_choice_idx[i:] = args.pad_token_idx # the remainder of the sequence
        cell_sentences[c, :] = torch.from_numpy(ordered_choice_idx)
        
    cell_sentences_pe = cell_sentences.long() # token indices
    
    return cell_sentences_pe, mask, longest_seq_len, cell_sentences


def sample_perturbed_cell_sentences(counts, batch_weights, dataset, args,
                          dataset_to_protein_embeddings,
                          dataset_to_chroms, 
                          dataset_to_starts, perturb_gene,
                          perturb_level=5):

    dataset_idxs = dataset_to_protein_embeddings[dataset] # get the dataset specific protein embedding idxs
    cell_sentences = torch.zeros((counts.shape[0], args.pad_length)) # init the cell representation as 0s
    mask = torch.zeros((counts.shape[0], args.pad_length)) # start of masking the whole sequence
    chroms = dataset_to_chroms[dataset] # get the dataset specific chroms for each gene
    starts = dataset_to_starts[dataset] # get the dataset specific genomic start locations for each gene
    all_gene_names = chroms.index.values

    longest_seq_len = 0 # we need to keep track of this so we can subset the batch at the end
    perturb_flag = []
    all_cell_sentences = []
    all_masks = []

    for case in ['unperturbed', 'perturbed']:
        
        ## First set the expression of the TF to zero
        perturb_idx = np.where(all_gene_names == perturb_gene)[0][0]
        batch_weights[0, perturb_idx] = 0

        for c, cell in enumerate(counts):
            
            if case == 'unperturbed':
                weights = batch_weights[c].numpy()
                weights = weights / sum(weights)  # RE NORM after mask

                # randomly choose the genes that will make up the sample, weighted by expression, with replacement
                choice_idx = np.random.choice(np.arange(len(weights)),
                                              size=args.sample_size, p=weights,
                                              replace=True)
            elif case == 'perturbed':
                choice_idx[-perturb_level:] = [perturb_idx]*perturb_level
                
            choosen_chrom = chroms[choice_idx] # get the sampled genes chromosomes
            # order the genes by chromosome
            chrom_sort = np.argsort(choosen_chrom)  
            choice_idx = choice_idx[chrom_sort]

            # sort the genes by start
            new_chrom = chroms[choice_idx]
            choosen_starts = starts[choice_idx]

            ordered_choice_idx = np.full((args.pad_length),
                                         args.cls_token_idx)  # start with cls
            # i= 0 first token is CLS
            i = 1  # continue on to the rest of the sequence with left bracket being assumed.
            # Shuffle the chroms now, there's no natural order to chromosomes
            uq_chroms = np.unique(new_chrom)
            np.random.shuffle(uq_chroms) # shuffle

            # This loop is actually just over one cell
            for chrom in uq_chroms:
                # Open Chrom token
                ordered_choice_idx[i] = int(chrom) + args.CHROM_TOKEN_OFFSET # 
                i += 1
                # now sort the genes by start order within the chroms
                loc = np.where(new_chrom == chrom)[0]
                sort_by_start = np.argsort(
                    choosen_starts[loc])  # start locations for this chromsome

                to_add = choice_idx[loc[sort_by_start]]
                ordered_choice_idx[i:(i + len(to_add))] = dataset_idxs[to_add]
                i += len(to_add)
                ordered_choice_idx[i] = args.chrom_token_right_idx # add the chrom sep again
                i += 1  # add the closing token again

            longest_seq_len = max(longest_seq_len, i)
            remainder_len = (args.pad_length - i)

            cell_mask = torch.concat((torch.ones(i),
                                      # pay attention to all of these tokens, ignore the rest!
                                      torch.zeros(remainder_len)))

            mask[c, :] = cell_mask

            ordered_choice_idx[i:] = args.pad_token_idx # the remainder of the sequence
            cell_sentences[c, :] = torch.from_numpy(ordered_choice_idx)
            
            if case == 'unperturbed':
                perturb_flag.append([0] * len(cell_sentences))
            elif case == 'perturbed':
                perturb_flag.append([1] * len(cell_sentences))
                
            all_cell_sentences.append(torch.clone(cell_sentences))
            all_masks.append(torch.clone(mask))

    all_cell_sentences_pe = [x.long() for x in all_cell_sentences] #
    # token indices
    
    return all_cell_sentences_pe, all_masks, longest_seq_len, \
        all_cell_sentences, perturb_flag