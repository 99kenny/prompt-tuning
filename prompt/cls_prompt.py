import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize

class ClsPrompt(nn.Module):
    def __init__(self, embed_dim=768, embedding_key='mean', prompt_key=False, pool_size=None,
                 top_k=None, batchwise_prompt=False, prompt_key_init='uniform', initial_prompt=None):
        super().__init__()

        self.embedding_key = embedding_key
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt   
        self.prompt = nn.Parameter(initial_prompt, requires_grad=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4*top_k)
        )    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None, is_train=True):
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
        
        batched_prompt_raw = self.mlp(cls_features)

        batch_size, top_k*length*c = batched_prompt_raw.shape
        
        
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

        out['prompt_idx'] = idx
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
