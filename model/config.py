chunk_length = 8 
dof = 23 
action_dim = 512 
max_seq_len = 64
tau = 1 
batch_size = 32
image_height = 256 
image_width = 256 


#transformer: 


context_length = 512    
D_model = action_dim
n_heads = 12 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ffn_mult = 8/3
hidden_dim = int(D_model * ffn_mult)
n_layers = 32
tau_embedding_dim = 512 
