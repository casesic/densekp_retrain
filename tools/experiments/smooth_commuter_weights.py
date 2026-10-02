"""Remove nearest-body assignment discontinuities on a garment's own graph."""


def main():
    import argparse
    import json
    from pathlib import Path
    import time
    import numpy as np
    from scipy.sparse import coo_matrix,diags

    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    rest=np.load(a.source/'clothing_rest_vertices.npy');faces=np.load(a.source/'clothing_faces.npy');initial=np.load(a.source/'skin_weights.npy')
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    length=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)
    weights=1/np.maximum(length,.002)
    graph=coo_matrix((np.r_[weights,weights],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(rest),len(rest))).tocsr()
    graph=diags(1/np.asarray(graph.sum(1)).ravel())@graph
    skin=initial.copy()
    for _ in range(24):skin=.5*skin+.5*(graph@skin)
    skin=np.maximum(skin,0);skin/=skin.sum(1,keepdims=True);skin=skin.astype(np.float32)
    np.save(a.output/'skin_weights.npy',skin)
    before=abs(initial[edges[:,0]]-initial[edges[:,1]]).sum(1);after=abs(skin[edges[:,0]]-skin[edges[:,1]]).sum(1)
    (a.output/'binding.json').write_text(json.dumps(dict(split='integration_only',source=str(a.source),
        method='24 half-step normalized inverse-length graph diffusion steps; 2mm edge-weight floor',
        geometry_changed=False,pose_independent=True,iterations=24,
        short_edge_threshold_m=.005,short_edge_L1_jump_threshold=.5,
        short_edge_jumps_before=int(((length<.005)&(before>.5)).sum()),short_edge_jumps_after=int(((length<.005)&(after>.5)).sum()),
        max_vertex_L1_change=float(abs(initial-skin).sum(1).max()),seconds=time.monotonic()-start),indent=2)+'\n')


if __name__ == "__main__":
    main()
