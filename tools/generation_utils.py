import torch, heapq, time
from torch import Tensor
from typing import Callable
from collections import OrderedDict
import os

def greedy_decode(transformer: Callable, **kwargs) -> Tensor:
    
    ## prepare inputs
    max_length = kwargs['max_length']
    inputs_embeds = kwargs.get('inputs_embeds') # batch x nwords x channel
    input_ids = kwargs.get('input_ids')
    vision_embeds = kwargs.get('vision_embeds')
    attention_mask = kwargs.get('attention_mask')
    vision_mask = kwargs.get('vision_mask')
    
    batch, _ = input_ids.shape
    
    ## prepare storage
    output_ids = torch.ones(batch, max_length).long().to(vision_embeds.device)
    output_ids = output_ids * kwargs['eos_token_id']
    
    ## prepare temporal storage of inputs
    # temporal_inputs = inputs_embeds
    finished_batchs = torch.zeros(batch).bool().to(vision_embeds.device)
    # embedding_layer = transformer.get_input_embeddings()
    for word_id in range(max_length):
        
        # step_output = transformer(
        #     inputs_embeds=temporal_inputs,
        #     output_attentions=False,
        # )

        step_output = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_embeds=vision_embeds,
            vision_mask=vision_mask,
            visual_prompt_embeds=kwargs['visual_prompt_embeds'],
            visual_prompt_mask=kwargs['visual_prompt_mask'],
            tokenizer=kwargs['tokenizer'],
            output_attentions=True,
        )
        
        attentions = step_output.get('attentions', None)
        
        next_word_id = step_output.logits[:, -1, :].argmax(-1)
        
        # check those finished sentences and overwrite
        finished_batchs |= (next_word_id == kwargs['eos_token_id'])
        next_word_id[finished_batchs] = kwargs['eos_token_id']
        
        output_ids[:, word_id] = next_word_id.long()    # (batch, )
        
        input_ids = torch.cat((input_ids, next_word_id.unsqueeze(1)), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones(batch, 1).to(attention_mask.device)), dim=1)
        # temporal_inputs = torch.cat((inputs_embeds, embedding_layer(output_ids[:, :word_id+1])), dim=1)
        
    return OrderedDict({'output_ids': output_ids.long(), 'attentions': attentions})


def beam_search_decode(transformer: Callable, **kwargs) -> Tensor:
    ## prepare inputs
    max_length = kwargs['max_length']
    
    # for safety issues
    assert kwargs['num_beams'] is not None, (
        'num_beams should not be provided if calling beam search!'
    )
    nbeams = kwargs['num_beams']
    input_ids = kwargs.get('input_ids')
    vision_embeds = kwargs.get('vision_embeds')
    attention_mask = kwargs.get('attention_mask')
    vision_mask = kwargs.get('vision_mask')
    visual_prompt_embeds = kwargs.get('visual_prompt_embeds')
    visual_prompt_mask = kwargs.get('visual_prompt_mask')
    
    batch, prefix_length = input_ids.shape
    channel = vision_embeds.shape[-1]
    
    # batch x nbeams x length x channel
    expanded_input_ids = input_ids.unsqueeze(1).repeat(1, nbeams, 1, 1)
    expanded_vision_embeds = vision_embeds.repeat(nbeams, 1, 1)
    expanded_attention_mask = attention_mask.unsqueeze(1).repeat(1, nbeams, 1)
    expanded_vision_mask = vision_mask.repeat(nbeams, 1)
    expanded_visual_prompt_embeds = visual_prompt_embeds.repeat(nbeams, 1, 1)
    expanded_visual_prompt_mask = visual_prompt_mask.repeat(nbeams, 1)
    
    ## prepare storage
    output_scores = torch.zeros(batch, nbeams).to(input_ids.device)
    output_ids = torch.ones(batch, nbeams, max_length).to(input_ids.device)
    output_ids = output_ids * kwargs['eos_token_id']
    batch_beam_results = OrderedDict({
        batch_id: [
            [float('-inf'), (float('-inf'), float('-inf')), None, None] \
                for b in range(nbeams)] \
                    for batch_id in range(batch)
    })
    
    for word_id in range(max_length):
        
        if word_id == 0:    # cold start for the first generation step
        
            step_output = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_embeds=vision_embeds,
                vision_mask=vision_mask,
                visual_prompt_embeds=visual_prompt_embeds,
                visual_prompt_mask=visual_prompt_mask,
                tokenizer=kwargs['tokenizer'],
            )
            # topk inds
            topk_scores, topk_inds = step_output.logits[:, -1, :].topk(
                k=nbeams, largest=True, dim=-1
            )   # batch x nbeams
            
            # store temporal scores for each beam
            output_ids[..., word_id] = topk_inds
            output_scores += torch.log_softmax(topk_scores, dim=-1)
            
        else:   # warm start from the previous step
            
            # batch x nbeams x word_id
            generated_words = output_ids[..., :word_id]
            
            # batch x nbeams x (length + word_id) x channel
            temporal_input_ids = torch.cat((expanded_input_ids, generated_words.long().unsqueeze(-2)), dim=-1)
            temporal_attention_mask = torch.cat((expanded_attention_mask, torch.ones(batch, nbeams, generated_words.shape[-1]).to(attention_mask.device)), dim=-1)
            
            step_output = transformer(
                input_ids=temporal_input_ids.reshape(batch * nbeams, prefix_length + word_id),
                attention_mask=temporal_attention_mask.reshape(batch * nbeams, prefix_length + word_id),
                vision_embeds=expanded_vision_embeds,
                vision_mask=expanded_vision_mask,
                visual_prompt_embeds=expanded_visual_prompt_embeds,
                visual_prompt_mask=expanded_visual_prompt_mask ,
                tokenizer=kwargs['tokenizer'],
                output_attentions=False,
            )
            last_word_logits = step_output.logits[:, -1, :].reshape(
                batch, nbeams, -1
            )   # batch x nbeams x nvocabs
            
            # beam_scores: batch x nbeams x nvocabs
            if word_id != max_length - 1:
                beam_scores = output_scores.unsqueeze(-1) + torch.log_softmax(
                    last_word_logits, dim=-1
                )
                
                output_scores, select_inds = beam_scores.reshape(batch, -1).topk(
                    k=nbeams, largest=True, dim=-1
                )
                # batch x k
                select_beam_id = select_inds // last_word_logits.shape[-1]
                select_word_id = select_inds % last_word_logits.shape[-1]
                
            else:
                
                # force ends of certain captions
                last_word_probs = torch.log_softmax(last_word_logits, dim=-1)
                output_scores += last_word_probs[..., kwargs['eos_token_id']]
                select_beam_id = \
                    torch.arange(nbeams).to(output_ids.device).unsqueeze(0).repeat(batch, 1)
                select_word_id = \
                    torch.ones_like(output_ids[..., -1]) * kwargs['eos_token_id']
            
            # gather generated beams
            output_ids = torch.gather(
                output_ids, 1, 
                select_beam_id.unsqueeze(-1).repeat(1, 1, max_length)
            )
            output_ids[..., word_id] = select_word_id
            
            ## ---- process the finished beams: batch x nbeams
            sentence_log_prob = output_scores / (word_id + 1)
            
            finished_batch, finished_beams = torch.where(
                select_word_id == kwargs['eos_token_id']
            )
            for batch_id, beam_id in zip(
                    finished_batch.cpu().tolist(), 
                    finished_beams.cpu().tolist()
                ):
                sentence = [
                    sentence_log_prob[batch_id, beam_id].cpu().tolist(),
                    (word_id, beam_id),
                    output_ids[batch_id, beam_id],          # max_length
                    sentence_log_prob[batch_id, [beam_id]]  # 1
                ]
                heapq.heappushpop(batch_beam_results[batch_id], sentence)
                
            
            # neglect the finished beam
            output_scores[select_word_id == kwargs['eos_token_id']] = -float('inf')
            
    ## final call, gather beam results from heaps
    output_ids = torch.cat([
        torch.cat(
            [
                beam_sentence.unsqueeze(0) \
                    for _, _, beam_sentence, _ in batch_beam_results[batch_id]
            ], dim=0
        ).unsqueeze(0) \
            for batch_id in range(batch)
    ], dim=0)   # batch x beam x max_length
    
    output_scores = torch.cat([
        torch.cat(
            [
                beam_log_prob.unsqueeze(0) \
                    for _, _, _, beam_log_prob in batch_beam_results[batch_id]
            ], dim=0
        ).unsqueeze(0) \
            for batch_id in range(batch)
    ], dim=0).squeeze(-1)   # batch x beam x 1
    
    return OrderedDict({
        'output_ids': torch.gather(
            output_ids.long(), 1, 
            output_scores.argmax(-1, keepdim=True).unsqueeze(1).repeat(1, 1, max_length)
        ).squeeze(1),
        'output_scores': output_scores,
        'beam_output_ids': output_ids.long()
    })
    

def generation(transformer: Callable, **kwargs):
    
    # parse keyword arguments, and assign default values
    kwargs['max_length'] = kwargs.get('max_length', 32)
    kwargs['early_stopping'] = kwargs.get('early_stopping', True)
    kwargs['num_beams'] = kwargs.get('num_beams', None)
    kwargs['eos_token_id'] = kwargs.get('eos_token_id', -1)
    kwargs['restore_prefix'] = kwargs.get('restore_prefix', False)
    
    input_ids  = kwargs.get('input_ids', None)
    inputs_embeds = kwargs.get('inputs_embeds', None)
    embedding_layer = transformer.get_input_embeddings()
    
    # if inputs_embeds is not None:
    #     assert input_ids is None, (
    #         'for safety issues, inputs_embeds is prior to input_ids!'
    #     )
    # elif input_ids is not None:
    #     kwargs['inputs_embeds'] = embedding_layer(input_ids)
    # else:
    #     raise NotImplementedError
    
    if kwargs['num_beams'] is None:
        # batch x max_length
        outputs = greedy_decode(transformer, **kwargs)
    else:
        outputs = beam_search_decode(transformer, **kwargs)
        
    ## post-processing, adding prefix if necessary
    if kwargs['restore_prefix'] is True:
        assert input_ids is not None, (
            'prefix could be only added when prefix ids is provided!'
        )
        outputs['output_ids'] = torch.cat(
            [input_ids, outputs['output_ids']], dim=-1
        )
    
    return outputs