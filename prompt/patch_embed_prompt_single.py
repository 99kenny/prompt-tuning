import torch
import torch.nn as nn
from torchvision import transforms
from utils import compute_similarity

class PatchEmbedPromptSingle(nn.Module):
    def __init__(self, img_size=224, length=5, embed_dim=768, embedding_key='mean', 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform', initial_prompt=None):
        super().__init__()
        self.img_size = img_size
        self.length = length
        self.embed_dim = embed_dim
        self.embedding_key = embedding_key
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.prompt = nn.Parameter(initial_prompt, requires_grad=True)
        self.frequency = torch.ones(pool_size).cuda()
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None, patch_embed=None):
        out = dict()
        
        if self.embedding_key == 'mean':
            x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'max':
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'cls':
            if cls_features is None:
                x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
            else:
                x_embed_mean = cls_features
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")
        
        similarity, prompt_norm, x_embed_norm = compute_similarity(self.prompt_key, x_embed_mean, self.frequency, True)
        
        if prompt_mask is None:
            _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
            if self.batchwise_prompt:
                prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                # Unless dimension is specified, this will be flattend if it is not already 1D.
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                    id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                major_prompt_id = prompt_id[major_idx] # top_k
                # expand to batch
                idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
        else:
            idx = prompt_mask # B, top_k
        
        for index in idx:
            self.frequency[index] = self.frequency[index] + 1
           
        batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
        # fix original code
        batch_size, top_k, c, h, w = batched_prompt_raw.shape
        
        resize = transforms.Compose([
            transforms.Resize(self.img_size)
        ])

        # divide into patches 
        batched_prompt = []
        for prompt_raw in batched_prompt_raw:
            batched_prompt.append(patch_embed(resize(prompt_raw))) #1, patch_num, embed_dim
        batched_prompt = torch.cat(batched_prompt, dim=0) # patch_num = 196 / 4 = 49
        # batched_prompt = batched_prompt.reshape(batch_size, top_k * self.length, c*16*16) # B, patch_num, C
        out['prompt_idx'] = idx

        # Debugging, return sim as well
        out['prompt_norm'] = prompt_norm
        out['x_embed_norm'] = x_embed_norm
        out['similarity'] = similarity

        # Put pull_constraint loss calculation inside
        batched_key_norm = prompt_norm[idx] # B, top_k, C
        out['selected_key'] = batched_key_norm
        x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
        sim = batched_key_norm * x_embed_norm # B, top_k, C
        reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

        out['reduce_sim'] = reduce_sim
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)
        
        return out
