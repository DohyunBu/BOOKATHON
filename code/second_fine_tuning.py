import tensorflow as tf
import torch
import transformers
from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast
from fastai.text.all import *
import fastai
import re
import pandas as pd
import os
from pathlib import Path

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>') 
model = AutoModelWithLMHead.from_pretrained("./fine_tuning2/first_model/")

path = [str(x) for x in Path("./전처리_데이터2/").glob("*.txt")]

single_string = ' '

for filename in path:
    with open(filename, "r", encoding='utf-8') as f:
        x=f.read()
    single_string += x + tokenizer.eos_token
lines = single_string

class TransformersTokenizer(Transform):
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def encodes(self, x): 
        toks = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.convert_tokens_to_ids(toks))
    def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))

#split data
train=lines[:-5]
test=lines[5:]
splits = [[0],[1]]

#init dataloader
tls = TfmdLists([train,test], TransformersTokenizer(tokenizer), splits=splits, dl_type=LMDataLoader)
batch,seq_len = 8,512
dls = tls.dataloaders(bs=batch, seq_len=seq_len)

class DropOutput(Callback):
    def after_pred(self): self.learn.pred = self.pred[0]
        
        
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), cbs=[DropOutput], metrics=Perplexity()).to_fp16()
#for para in learn.model.parameters():
#    para.requires_grad = False
#for name, param in learn.model.named_parameters():
#    print(name)
#    if "transformer.wpe" in name:
#        param.requires_grad = True

#for i, m in enumerate(model.transformer.h):
#    if i<=3:
#        for parameter in m.parameters():
#            parameter.requires_grad = True

#for parameter in model.transformer.ln_f.parameters():
#    parameter.requires_grad = True

#for parameter in model.transformer.wpe.parameters():
#    parameter.requires_grad = True

print(learn.summary())
lr= learn.lr_find()
print(lr)
learn.fit_one_cycle(5, lr.valley)

learn.model.save_pretrained("./fine_tuning3/first_model/")