import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import transforms
from utils import compute_similarity

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(MLPEncoder, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.FC_mean  = nn.Linear(hidden_dim, input_dim)
        self.FC_var   = nn.Linear(hidden_dim, input_dim)
        self.training = True
        
    def forward(self, x):
        x = self.net(x)
        mean     = self.FC_mean(x)
        log_var  = self.FC_var(x)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        return mean, log_var

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x_hat = self.net(x)
        return x_hat

class Vae(nn.Module):
    def __init__(self, encoder, decoder, length=5, embed_dim=768):
        super(Vae, self).__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.encoder = encoder
        self.decoder = decoder        
        
        print(f'Prompt Generator Initialized - length : {self.length}, embed_dim = {embed_dim}')
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).cuda()
        z = mean + var * epsilon
        return z
    
    def inference(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mean, var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * var))
        x_hat = self.decoder(z)
        return x_hat, mean, var

class GeneratorPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', pool_size=None, top_k=None, batchwise_prompt=False, img_size=224,
                 prompt_key_init='uniform',input_dim=768, latent_dim=200, hidden_dim=400, initial_image=None,original_model=None):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.embedding_key = embedding_key        
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.img_size = img_size
        self.original_model = original_model
        self.frequency = torch.ones(pool_size).cuda()
        self.generator = Vae(MLPEncoder(input_dim, hidden_dim, latent_dim), 
                          MLPDecoder(latent_dim, hidden_dim, input_dim), 
                          length=length, embed_dim=embed_dim)
        prompt_key = True
        prompt_pool_shape = (pool_size, length, embed_dim)
        # self.prompt = nn.Parameter(initial_image,requires_grad=True)
        if True:
            self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
            nn.init.uniform_(self.prompt, -1, 1)
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
    
    def reinit_freq(self,):
        self.frequency = torch.ones(self.pool_size).cuda()
    
    def forward(self, is_training, x_embed, prompt_mask=None, cls_features=None):
        # resize = transforms.Compose([
        #     transforms.Resize(self.img_size)
        # ])
        out = dict()
        
        x_embed_mean = cls_features
        # # prompt_key = self.original_model(resize(self.prompt))['pre_logits']
        prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C
        similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
        
        similarity, prompt_norm, x_embed_norm = compute_similarity(prompt_norm, x_embed_mean, self.frequency, is_training)
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
            
        if is_training:
            for index in idx:
                self.frequency[index] = self.frequency[index] + 1
            
            # B, embed_dim
            synthesized_features, mean, var = self.generator(cls_features)

        else:
            synthesized_features = self.generator.inference(torch.randn(cls_features.shape).cuda())
            mean = None
            var = None
        batched_prompt_raw = self.prompt[idx] # B, t, c, h, w
        
        batch_size, top_k, l, dim = batched_prompt_raw.shape 
        
        # synthesized feature
        batched_prompt_logits = batched_prompt_raw.reshape(batch_size, top_k * self.length, dim) # B, top_k * length, C
        # original feature
        
        out['mean'] = mean
        out['var'] = var
        out['cls_features'] = cls_features # B, embed_dim
        out['synthesized_features'] = synthesized_features
        expanded_cls = synthesized_features.unsqueeze(1).expand(batch_size, top_k * self.length, dim)
        batched_prompt = batched_prompt_logits + expanded_cls
        batched_prompt = torch.cat([batched_prompt, cls_features.unsqueeze(1)], dim=1)

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
