"""Text-only use of a frozen native backbone, outside manipulation inference.

The caller owns explicit call accounting. No image, state, outcome or task ID
enters this encoder. Static instruction caches never alter the native sampler.
"""
import math
import numpy as np


def pi05_text_features(model,tokenizer,text):
    import torch
    if not isinstance(text,str) or not text.strip():raise ValueError('Task instruction required')
    cleaned=text.strip().replace('_',' ').replace('\n',' ')
    ids=tokenizer.encode(cleaned,add_bos=True)
    if not 2<=len(ids)<=200:raise ValueError('No empty/truncated instruction encoding')
    native=model.paligemma_with_expert
    parameter=next(native.parameters());tokens=torch.tensor([ids],dtype=torch.long,device=parameter.device)
    with torch.inference_mode():
        embeds=native.embed_language_tokens(tokens)
        embeds=embeds*math.sqrt(embeds.shape[-1])
        mask=torch.zeros((1,1,len(ids),len(ids)),dtype=embeds.dtype,device=embeds.device)
        positions=torch.arange(len(ids),device=embeds.device)[None]
        (hidden,_),_=native(attention_mask=mask,position_ids=positions,inputs_embeds=[embeds,None],use_cache=False)
        # BOS participates in attention but is not a task-word pooling member.
        value=hidden[0,1:].float().mean(0).cpu().numpy().copy()
    if value.ndim!=1 or not np.isfinite(value).all():raise ValueError('Invalid native text-only representation')
    return value,dict(token_count=len(ids),pooled_token_count=len(ids)-1,images=0,states=0,action_expert_calls=0)
