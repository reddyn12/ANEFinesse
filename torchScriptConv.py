import os
from contextlib import nullcontext
import sys
import torch
# import tiktoken
from customModel import GPTConfig, GPT
import customToken
import coremltools as ct
import numpy as np

init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'custOut' # ignored if init_from is not 'resume'

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()

s = "1 . Nf3 d5 2 . b3 Bg4 3 . e3 Nd7 4 . Bb2 e6 5 . Be2 Ngf6 6 . O-O Bd6 7 . c4 c6 8 . h3 Bh5 9 . Nc3 Qe7 10 . d4 O-O 11 . Bd3 Bg6 12 . Qc2 Rfe8 13 . Rad1 Rad8 14 . e4 dxe4 15 . Nxe4 Nxe4 16 . Bxe4 Bxe4 17 . Qxe4 f5 18 . Qe3 c5 19 . Rfe1 cxd4 20 . Nxd4 Be5 21 . Nc6 Bxb2 22 . Nxe7 + Rxe7 23 . Qxa7 Ba1 24 . Rxa1"

ids = customToken.tokenize(s)
print("ids: ", ids)
idx = torch.tensor(ids)[None, ...]
print("idx: ", idx)

# sys.exit()
# if the sequence context is growing too long we must crop it at block_size
idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
# a= model(idx_cond)
# print(a)
traced = torch.jit.trace(model, idx_cond)
# print(idx[0].tolist())

# print(idx.shape)
# sys.exit()

ane_mlpackage_obj = ct.convert(
    traced,
    convert_to="mlprogram",
    # inputs=[idx[0].tolist()],
    inputs=[
        ct.TensorType(
                f"input_1",
                    shape=idx.shape,
                    dtype=np.int32,
                ) 
            ],
            compute_units=ct.ComputeUnit.ALL,
)
out_path = "testChess.mlpackage"
ane_mlpackage_obj.save(out_path)