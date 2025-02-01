import os
import numpy as np
import clip
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='DisNet')
    parser.add_argument('--out_dir', type=str, default='./', help='specify the output directory')
    parser.add_argument('--text_prompt', type=str, default='foo', help='specify the input text prompt')
    args = parser.parse_args()
    return args

def main():
    args = get_parser()
    text_prompt = args.text_prompt
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # "ViT-L/14@336px" # the big model that OpenSeg uses
    print('Loading the CLIP model...')
    clip_pretrained, _ = clip.load("ViT-L/14@336px", device='cpu', jit=False)
    #clip_pretrained, _ = clip.load("ViT-L/14@336px", device='cuda', jit=False)
    print('Finish loading.')
    print()

    # generate token
    text = clip.tokenize([text_prompt])
    #text = clip.tokenize([text_prompt]).cuda()
    text_features = clip_pretrained.encode_text(text)
    # normalization
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    #print(text_features)
    # save features
    np.save(os.path.join(out_dir, '{}.npy'.format(text_prompt)), text_features.detach().cpu().numpy())
    print()
    print('CLIP feature of "{}" is saved to {}'.format(text_prompt, os.path.join(out_dir, '{}.npy'.format(text_prompt))))



if __name__ == '__main__':
    main()
