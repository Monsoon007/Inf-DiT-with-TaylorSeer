/root/miniconda3/bin/conda run -n DiT --no-capture-output python /root/Inf-DiT-with-TaylorSeer/generate_t2i_sr.py --input-type txt --input-path /root/Inf-DiT-with-TaylorSeer/image/Input.txt --inference_type full --block_batch 4 --experiment-name generate --mode inference --inference-batch-size 1 --image-size 512 --input-time adaln --nogate --no-crossmask --bf16 --num-layers 28 --vocab-size 1 --hidden-size 1280 --num-attention-heads 16 --hidden-dropout 0. --attention-dropout 0. --in-channels 6 --out-channels 3 --cross-attn-hidden-size 640 --patch-size 4 --config-path configs/text2image-sr.yaml --max-sequence-length 256 --layernorm-epsilon 1e-6 --layernorm-order pre --model-parallel-size 1 --tokenizer-type fake --random-position --qk-ln --out-dir samples --network ckpt/mp_rank_00_model_states.pt --round 32 --init_noise --image-condition --vector-dim 768 --re-position --cross-lr --seed 42 --infer_sr_scale 4 --guider TaylorSeerGuider --guiderscale 0
[2025-03-31 18:06:36,144] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-03-31 18:06:38,714] [WARNING] No training data specified
[2025-03-31 18:06:38,714] [WARNING] No train_iters (recommended) or epochs specified, use default 10k iters.
[2025-03-31 18:06:38,714] [INFO] using world size: 1 and model-parallel size: 1
[2025-03-31 18:06:38,714] [INFO] > padded vocab (size: 1) with 127 dummy tokens (new size: 128)
[2025-03-31 18:06:38,716] [INFO] [RANK 0] > initializing model parallel with size 1
[2025-03-31 18:06:38,717] [INFO] [RANK 0] You didn't pass in LOCAL_WORLD_SIZE environment variable. We use the guessed LOCAL_WORLD_SIZE=1. If this is wrong, please pass the LOCAL_WORLD_SIZE manually.
Loading network from "ckpt/mp_rank_00_model_states.pt"...
[2025-03-31 18:06:38,718] [INFO] [RANK 0] building DiffusionEngine model ...
--------use random position--------
warning: cross_attn_hidden_size is set but is_decoder is False
 [DiffusionEngine] ✅ Overriding guider: dit.sampling.guiders.TaylorSeerGuider
[DiffusionEngine] ✅ Overriding guider scale: 0.0
--------use qk_ln--------
 [DiffusionEngine] Sampler config: {'target': 'dit.sampling.samplers.ConcatSRHeunEDMSampler', 'params': {'num_steps': 20, 'discretization_config': {'target': 'dit.sampling.discretizers.EDMDiscretization', 'params': {'sigma_min': 0.002, 'sigma_max': 40}}, 'guider_config': {'target': 'dit.sampling.guiders.IdentityGuider'}}}
[DiffusionEngine] ✅ Setting guider in sampler config: dit.sampling.guiders.TaylorSeerGuider
[DiffusionEngine] Updated sampler config: {'target': 'dit.sampling.samplers.ConcatSRHeunEDMSampler', 'params': {'num_steps': 20, 'discretization_config': {'target': 'dit.sampling.discretizers.EDMDiscretization', 'params': {'sigma_min': 0.002, 'sigma_max': 40}}, 'guider_config': {'target': 'dit.sampling.guiders.IdentityGuider'}}, 'guider_config': {'target': 'dit.sampling.guiders.TaylorSeerGuider'}}
[DiffusionEngine] ✅ Setting guider scale in sampler config: 0.0
[DiffusionEngine] Sampler initialized: <dit.sampling.samplers.ConcatSRHeunEDMSampler object at 0x7f8294383fe0>
[DiffusionEngine] Sampler guider config: None
[DiffusionEngine] Guider already set or no guider config available
[2025-03-31 18:07:22,613] [INFO] [RANK 0]  > number of parameters on model parallel rank 0: 1096323441
INFO:sat:[RANK 0]  > number of parameters on model parallel rank 0: 1096323441
/root/Inf-DiT-with-TaylorSeer/generate_t2i_sr.py:113: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  data = torch.load(args.network, map_location='cpu')
Loading Fished!
rank: 0 world_size: 1
0it [00:00, ?it/s]
  0%|                                                    | 0/20 [00:00<?, ?it/s]没有启用成功taylorguider
[FLOPs] Warning: FLOPs estimation failed. 'function' object has no attribute '_modules'
没有启用成功taylorguider

  5%|██▏                                         | 1/20 [00:00<00:14,  1.32it/s]没有启用成功taylorguider
没有启用成功taylorguider

 10%|████▍                                       | 2/20 [00:01<00:13,  1.38it/s]没有启用成功taylorguider
没有启用成功taylorguider

 15%|██████▌                                     | 3/20 [00:02<00:12,  1.40it/s]没有启用成功taylorguider
没有启用成功taylorguider

 20%|████████▊                                   | 4/20 [00:02<00:11,  1.41it/s]没有启用成功taylorguider
没有启用成功taylorguider

 25%|███████████                                 | 5/20 [00:03<00:10,  1.41it/s]没有启用成功taylorguider
没有启用成功taylorguider

 30%|█████████████▏                              | 6/20 [00:04<00:09,  1.42it/s]没有启用成功taylorguider
没有启用成功taylorguider

 35%|███████████████▍                            | 7/20 [00:04<00:09,  1.42it/s]没有启用成功taylorguider
没有启用成功taylorguider

 40%|█████████████████▌                          | 8/20 [00:05<00:08,  1.42it/s]没有启用成功taylorguider
没有启用成功taylorguider

 45%|███████████████████▊                        | 9/20 [00:06<00:07,  1.42it/s]没有启用成功taylorguider
没有启用成功taylorguider

 50%|█████████████████████▌                     | 10/20 [00:07<00:07,  1.42it/s]没有启用成功taylorguider
没有启用成功taylorguider

 55%|███████████████████████▋                   | 11/20 [00:07<00:06,  1.42it/s]没有启用成功taylorguider
没有启用成功taylorguider

 60%|█████████████████████████▊                 | 12/20 [00:08<00:05,  1.42it/s]没有启用成功taylorguider
没有启用成功taylorguider

 65%|███████████████████████████▉               | 13/20 [00:09<00:04,  1.42it/s]没有启用成功taylorguider
没有启用成功taylorguider

 70%|██████████████████████████████             | 14/20 [00:09<00:04,  1.42it/s]没有启用成功taylorguider
没有启用成功taylorguider

 75%|████████████████████████████████▎          | 15/20 [00:10<00:03,  1.42it/s]没有启用成功taylorguider
没有启用成功taylorguider

 80%|██████████████████████████████████▍        | 16/20 [00:11<00:02,  1.42it/s]没有启用成功taylorguider
没有启用成功taylorguider

 85%|████████████████████████████████████▌      | 17/20 [00:12<00:02,  1.42it/s]没有启用成功taylorguider
没有启用成功taylorguider

 90%|██████████████████████████████████████▋    | 18/20 [00:12<00:01,  1.41it/s]没有启用成功taylorguider
没有启用成功taylorguider

 95%|████████████████████████████████████████▊  | 19/20 [00:13<00:00,  1.41it/s]没有启用成功taylorguider

100%|███████████████████████████████████████████| 20/20 [00:13<00:00,  1.45it/s]
save to samples/n_sr.png
[FLOPs] Total test-time FLOPs: 0.00 GFLOPs
1it [00:14, 14.40s/it]
[rank0]:[W331 18:07:44.865016038 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())

Process finished with exit code 0