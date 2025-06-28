import time

out_dir = 'out-wikitext2'
eval_interval = 10
eval_iters = 50
wandb_log = True # feel free to turn on
wandb_project = 'nano-moe'


dataset = 'wikitext2'
init_from = 'resume' # this is the largest GPT-2 model
wandb_run_name = 'ft-moe-topk-' + dataset + '-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
# model/moe settings
n_exp = 8
top_k = 2
use_batch_topk = False
use_aux_loss = True
aux_loss_weight = 0.01
use_router_z_loss = True
router_z_loss_weight = 0.001
use_noisy_top_k = False
train_capacity = 1.25
eval_capacity = 2.0
stride = 2
use_switch_tfm_init = True
router_use_full_prec = True

# use smaller GPT model
n_layer = 6
n_head = 6
n_embd = 384

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 2
gradient_accumulation_steps = 32
max_iters = 500



# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
ckpt_path = '/home/ubuntu/tiansheng/nanoMoE-ts/out/small_pre_trained_moe_ckpt.pt'
