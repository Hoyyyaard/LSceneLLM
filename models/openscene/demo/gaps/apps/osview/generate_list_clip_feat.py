import os
import numpy as np
import clip
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='DisNet')
    parser.add_argument('--in_file', type=str, default='./class_names.txt', help='specify the input class names list')
    parser.add_argument('--out_file', type=str, default='./class_features.npy', help='specify the output class features file')
    args = parser.parse_args()
    return args

def main():
  args = get_parser()
  in_file = args.in_file
  out_file = args.out_file

  # "ViT-L/14@336px" # the big model that OpenSeg uses
  print('Loading CLIP model...')
  clip_pretrained, _ = clip.load("ViT-L/14@336px", device='cpu', jit=False)
  #clip_pretrained, _ = clip.load("ViT-L/14@336px", device='cuda', jit=False)

  # Initialize matrix
  count = 0
  file = open(in_file, 'r')
  lines = file.readlines()
  nclasses = len(lines)
  matrix = np.zeros((nclasses, 768), float)

  # Read class names and compute features
  print('Reading class names from ' + in_file)
  for line in lines:
    class_name = line.strip()
    print('  ' + class_name)

    # generate token
    text = clip.tokenize([class_name])
    #text = clip.tokenize([class_name]).cuda()

    # generate features
    features = clip_pretrained.encode_text(text)
    
    # normalize features
    features = features / features.norm(dim=-1, keepdim=True)

    # add features to matrix
    matrix[count] = features.detach().cpu().numpy()

    # increment counter
    count = count + 1

  # Print matrix to output file
  print('Writing features to ' + out_file)
  np.save(out_file.format(float), matrix)

if __name__ == '__main__':
    main()
