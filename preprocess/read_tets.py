import os
import numpy as np
import torch

def main():
    device = 'cpu'
    grid_res = 90
    base_dir = BASE_DIR
    tets_file = 'data/tets/%d_compress.npz' % (grid_res)
    tets_path = os.path.join(base_dir, tets_file)
    tets = np.load(tets_path)

    verts = torch.from_numpy(tets['vertices']).float().to(device)
    print('verts.shape ', verts.shape)
    print('x min ', verts.min(dim=0)[0], 'x max ', verts.max(dim=0)[0])

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
