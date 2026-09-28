import torch
import torch.nn as nn
import torch.nn.functional as F

class SASRecDualEncoder(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=512, num_layers=2, num_heads=4, max_seq_len=20, dropout=0.2):
        super(SASRecDualEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Project input embeddings to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional Encoding
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        # PyTorch's nested-tensor padding fast path is not implemented on MPS.
        # Disabling it keeps the Transformer computation on the Apple GPU.
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        
        # Project back to input dimension to match target embedding space
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, seq_embeddings, lengths):
        """
        seq_embeddings: (batch_size, seq_len, input_dim)
        lengths: (batch_size,) actual lengths of sequences before padding
        """
        batch_size, seq_len, _ = seq_embeddings.size()
        
        # 1. Project to hidden dim
        x = self.input_projection(seq_embeddings) # (B, L, H)
        
        # 2. Add positional encodings
        positions = torch.arange(seq_len, device=seq_embeddings.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.position_embedding(positions)
        
        x = x + pos_emb
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # 3. Create padding mask for Transformer
        # True means masked out (ignore padding)
        mask = torch.arange(seq_len, device=seq_embeddings.device).unsqueeze(0).expand(batch_size, seq_len)
        mask = mask >= lengths.unsqueeze(1) # (B, L)
        
        # 4. Transformer
        # We need a causal mask to prevent looking into the future
        # Match the Boolean padding-mask dtype and avoid the deprecated
        # float-mask/Boolean-mask combination.
        causal_mask = torch.triu(
            torch.ones(
                (seq_len, seq_len), dtype=torch.bool,
                device=seq_embeddings.device
            ),
            diagonal=1
        )
        
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=mask) # (B, L, H)
        
        # 5. Get the output corresponding to the LAST valid track in the sequence
        # out shape: (B, L, H)
        # We want out[b, lengths[b]-1, :]
        last_indices = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, self.hidden_dim)
        last_out = torch.gather(out, 1, last_indices).squeeze(1) # (B, H)
        
        # 6. Project back to embedding dimension
        pred_embedding = self.output_projection(last_out) # (B, input_dim)
        
        # 7. Normalize predicted embedding to use Cosine Similarity natively
        pred_embedding = F.normalize(pred_embedding, p=2, dim=1)
        
        return pred_embedding

def get_contrastive_loss(pred_embeddings, target_embeddings, temperature=0.07):
    """
    InfoNCE Loss (Contrastive)
    pred_embeddings: (B, D) normalized predicted embeddings
    target_embeddings: (B, D) normalized ground truth next-track embeddings
    """
    # Cosine similarity matrix: (B, B)
    # Diagonal represents positive pairs, off-diagonal represents negative pairs
    logits = torch.matmul(pred_embeddings, target_embeddings.t()) / temperature
    
    # Labels are just the diagonal index
    labels = torch.arange(logits.size(0), device=logits.device)
    
    loss = F.cross_entropy(logits, labels)
    return loss
