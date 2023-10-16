# config for training custom GPT2-small (45M)

wandb_log = True
wandb_project = 'wikipedia'
wandb_run_name = 'gpt2-small'
dataset = 'wikipedia'

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 1 GPUs = 61,440
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 1

# this makes total number of tokens be 3B
max_iters = 50000
lr_decay_iters = 50000

# GPT model
n_layer = 6
n_head = 8
n_embd = 512
# dropout = 0.2

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10
out_dir = 'out-wikipedia-small'

# weight decay
weight_decay = 1e-1
learning_rate = 1e-3  # with baby networks can afford to go a bit higher
min_lr = 1e-4  # learning_rate / 10 usually

# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
