"""Shared Qwen3.5/Wan2.2 training arguments for scene-understanding tasks."""

def training_arguments(config, experiment, data_yaml, output, run_name, gpu_count, max_steps=None):
    m, t, d = config["model"], config["training"], config["data"]
    if t["global_batch"] % (gpu_count*t["micro_batch"]):
        raise ValueError("Global batch must divide by GPU count * micro batch")
    if t["precision"] != "bf16" or t["torch_compile"]:
        raise ValueError("This entry currently requires bf16 and torch_compile=false")
    args = dict(model_name_or_path=m["initial_checkpoint"], version="qwen35", data_path=str(data_yaml),
        vision_tower=m["vision_checkpoint"], image_folder="data", video_folder="data", embodiedscan_folder="data/embodiedscan/",
        mm_tunable_parts=t["tune"], mm_vision_tower_lr=t["vision_learning_rate"], mm_projector_type="mlp2x_gelu",
        mm_vision_select_layer=-2, mm_use_im_start_end=False, mm_use_im_patch_token=False,
        image_aspect_ratio="anyres_max_9", image_grid_pinpoints="(1x1),...,(6x6)", mm_patch_merge_type="spatial_unpad",
        bf16=True, tf32=True, output_dir=str(output), run_name=run_name, num_train_epochs=t["epochs"],
        max_steps=t["max_steps"] if max_steps is None else max_steps, report_to="none",
        per_device_train_batch_size=t["micro_batch"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=t["global_batch"]//(gpu_count*t["micro_batch"]), eval_strategy="no",
        save_strategy="steps", save_steps=t["save_steps"], save_total_limit=2,
        learning_rate=t["learning_rate"], foresight_lr=t.get("foresight_learning_rate", t["learning_rate"]), weight_decay=t["weight_decay"],
        optim=t["optimizer"], adam_beta1=t["adam_beta1"], adam_beta2=t["adam_beta2"],
        adam_epsilon=t["adam_epsilon"], max_grad_norm=t["max_grad_norm"],
        warmup_ratio=t["warmup_ratio"], lr_scheduler_type=t["scheduler"], logging_steps=1,
        model_max_length=t["model_max_length"], gradient_checkpointing=t["gradient_checkpointing"],
        dataloader_num_workers=t["data_workers"], lazy_preprocess=True, torch_compile=False,
        dataloader_drop_last=False, mm_newline_position="grid", add_spatial_instruction=True,
        force_sample=True, mm_spatial_pool_stride=2, world_position_embedding_type="avg-discrete-sin3d",
        object_feature_type="patch14-pe", group_by_task_length=False, frame_sampling_strategy="uniform",
        frames_upbound=d["frame_count"], use_generative_feature=True, generative_feat_dim=m["generative_feature_dim"],
        generative_projector_type="mlp2x_gelu", generative_feature_source="offline", generative_model_id=d["feature_id"],
        generative_cache_max_items=1, generative_profile=False, use_feat_fusion=True, feature_fusion_method="token_gated",
        seed=config["seed"], data_seed=config["seed"], foresight_enabled=experiment["queries_enabled"],
        foresight_num_queries=m.get("queries", 32), foresight_context_dim=m.get("wan_context_dim", 4096),
        foresight_video_loss_weight=experiment["video_loss_weight"], foresight_wan_checkpoint=m["wan_checkpoint"],
        foresight_video_frames=m.get("teacher_frames", 5), foresight_video_height=m.get("teacher_height", 256),
        foresight_video_width=m.get("teacher_width", 320), foresight_flow_shift=m.get("flow_shift", 5.0),
        foresight_wan_gradient_checkpointing=m.get("teacher_gradient_checkpointing", True))
    if m.get("ground_head_type") is not None:
        args["ground_head_type"] = m["ground_head_type"]
    if t["deepspeed"]:
        args["deepspeed"] = t["deepspeed"]
    return [value for key, item in args.items() for value in ("--"+key, str(item))]
