import os

TRANSFORMER_DECODER={'depth': 6,
                    'heads': 8,
                    'mlp_dim': 1024,
                    'dim_head': 64,
                    'dropout': 0.0,
                    'emb_dropout': 0.0,
                    'norm': 'layer',
                    'context_dim': 1280}

NUM_JOINTS = 44

NUM_DENSEKP_SMPL = 138

VITPOSE_BACKBONE = os.environ.get(
    'DENSEKP_VITPOSE_BACKBONE',
    'assets_external/vitpose_backbone.pth',
)
