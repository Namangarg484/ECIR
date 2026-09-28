import torch
import torch.nn as nn
import torch.nn.functional as F

class VCEModel(nn.Module):
    """
    Variational Centroid Expansion (VCE) Model.
    
    Instead of heuristic averaging of anchor tracks, this model:
    1. Learns to attend over anchor tracks using the query as context.
    2. Estimates the uncertainty (kappa) of the resulting semantic intent.
    3. Perturbs the centroid along random unit tangent directions.
       This is NOT an exact von Mises-Fisher sampler or a calibrated posterior.
    """
    def __init__(self, embed_dim=1024, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Cross-attention: Query attends to Anchor Tracks
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        
        # FFN for the attended representation
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Uncertainty Predictor (predicts log(kappa))
        self.kappa_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # Scale to ensure kappa is within reasonable bounds for vMF
        self.max_kappa = 500.0
        
        # Learned gate to restrict angular displacement
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, query_emb, anchor_embs, anchor_mask=None):
        """
        query_emb: (batch_size, embed_dim)
        anchor_embs: (batch_size, num_anchors, embed_dim)
        anchor_mask: (batch_size, num_anchors) bool tensor where True means padded/ignored
        
        Returns:
            mu: (batch_size, embed_dim) L2-normalized attended centroid
            kappa: (batch_size, 1) Concentration parameter
        """
        # Reshape query for attention (batch_size, 1, embed_dim)
        q = query_emb.unsqueeze(1)
        
        # Cross Attention
        # attn_output: (batch_size, 1, embed_dim)
        attn_output, attn_weights = self.attn(
            query=q,
            key=anchor_embs,
            value=anchor_embs,
            key_padding_mask=anchor_mask
        )
        
        # Intermediate representation for FFN
        h = self.layer_norm1(q + attn_output)
        ffn_output = self.ffn(h)
        
        # Compute the neural shift
        shift = self.layer_norm2(attn_output + ffn_output)
        shift = shift.squeeze(1)
        shift = F.normalize(shift, p=2, dim=-1)  # unit norm shift
        q_squeezed = q.squeeze(1)
        
        # Apply learned gate
        gate = torch.sigmoid(self.residual_gate)
        gate = gate * 0.2  # clamp max angular budget to ~0.2 radians
        x = q_squeezed + gate * shift
        
        # The direction is the L2 normalized output
        mu = F.normalize(x, p=2, dim=-1)
        
        # Predict log(kappa)
        log_kappa = self.kappa_predictor(x)
        
        # Map log_kappa to a stable positive range: 0 to max_kappa
        kappa = self.max_kappa * torch.sigmoid(log_kappa) + 1.0
        kappa = torch.clamp(kappa, min=10.0, max=500.0)
        
        return mu, kappa, attn_weights

    def sample_vmf(self, mu, kappa, n_samples=1):
        """
        Legacy API name: fixed-radius tangent-sphere perturbation, NOT exact vMF.
        The polar angle is approximately atan(1/sqrt(kappa)); only the tangent
        direction is random. Kept for old checkpoint/caller compatibility.
        
        mu: (batch_size, embed_dim) directional mean
        kappa: (batch_size, 1) concentration parameter
        n_samples: int, number of samples per batch item
        
        Returns:
            samples: (batch_size, n_samples, embed_dim)
        """
        batch_size, dim = mu.shape
        
        # A simple approximation for high dimensions and large kappa:
        # Sample from isotropic Gaussian in the tangent plane, then project back to sphere.
        # This is a scalable approximation suitable for retrieval expansion.
        
        # Generate random directions: (batch_size, n_samples, dim)
        epsilon = torch.randn(batch_size, n_samples, dim, device=mu.device)
        
        # Project out the component along mu to get vectors in the tangent space
        # mu is (batch_size, dim), epsilon is (batch_size, n_samples, dim)
        mu_expanded = mu.unsqueeze(1) # (batch_size, 1, dim)
        
        # Dot product: (batch_size, n_samples, 1)
        dot_product = torch.sum(epsilon * mu_expanded, dim=2, keepdim=True)
        
        # Tangent vector: (batch_size, n_samples, dim)
        tangent = epsilon - dot_product * mu_expanded
        tangent = F.normalize(tangent, p=2, dim=2)
        
        # The variance is roughly inversely proportional to kappa
        # (For exact vMF, we'd sample W from the marginal distribution, but here we approximate)
        sigma = 1.0 / torch.sqrt(kappa.unsqueeze(1) + 1e-6) # (batch_size, 1, 1)
        
        # Blend mu and tangent (small angle approximation)
        samples = mu_expanded + sigma * tangent
        
        # Re-normalize to ensure they lie on the unit hypersphere
        samples = F.normalize(samples, p=2, dim=2)
        
        return samples
