import argparse

from train import run_training
from eval import run_eval


# ============================================================================
# Single entry point for the DINOv3 + T5/SigLIP2 + Q-Former geolocalization
# pipeline. --mode selects training or evaluation; all the actual logic
# lives in train.py / eval.py respectively. This file only wires CLI flags
# to run_training() / run_eval().
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="GeoDinoSiglipQFormer -- train or evaluate")

    # Selects whether to run training or evaluation.
    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval"])

    # Path to the JSON config file used to build the model, dataset, losses, and training schedule.
    parser.add_argument("--config", type=str, default="configs/geo_dino_t5_qformer.json")

    # Optional CUDA device index. If omitted, use CUDA's default device (or CPU when CUDA is unavailable).
    parser.add_argument("--gpu-id", "--gpu_id", dest="gpu_id", type=int, default=None,
                        help="CUDA device index, e.g. 0 or 1")

    # Path to a Stage 1 checkpoint to resume from, skipping Stage 1 entirely and going straight
    # into Stage 2 (global epoch numbering continues from checkpoint_epoch + 1). Only used in --mode train.
    parser.add_argument(
        "--resume-from-stage1", type=str, default=None,
        help="train mode only: path to a Stage 1 checkpoint. Skips Stage 1, freezes it, and runs "
             "Stage 2 for exactly (total_epochs - stage1_epochs) epochs."
    )

    # Path to the saved checkpoint. Required only for --mode eval.
    parser.add_argument("--checkpoint", type=str, default=None, help="required for --mode eval")

    # Evaluation stage. Use 1 for global retrieval only, 2 for global retrieval plus region voting rerank.
    parser.add_argument("--stage", type=int, default=None, choices=[1, 2], help="required for --mode eval")

    # Number of global-retrieval candidates to keep before Stage 2 reranking.
    parser.add_argument("--rerank-topk", type=int, default=100)

    # Batch size for initial query/satellite encoding during evaluation.
    parser.add_argument("--eval-batch-size", type=int, default=64)

    # Number of queries processed at once during Stage 2 region voting reranking.
    parser.add_argument("--gather-batch-size", type=int, default=16)

    # Number of candidate satellites processed at once per query during Stage 2 reranking.
    parser.add_argument("--candidate-chunk-size", type=int, default=100)

    parser.add_argument(
        "--fusion-lambda",
        type=float,
        default=0.3,
        help="Weight on z-scored Stage 2 region-voting score during fused reranking. "
            "0.0 = pure global ranking; 0.3 is the default."
    )

    args = parser.parse_args()

    if args.mode == "train":
        run_training(args.config, resume_from_stage1=args.resume_from_stage1, gpu_id=args.gpu_id)

    elif args.mode == "eval":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for --mode eval")
        if args.stage is None:
            parser.error("--stage is required for --mode eval (1 = global only, 2 = global + region voting rerank)")

        run_eval(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            stage=args.stage,
            rerank_topk=args.rerank_topk,
            eval_batch_size=args.eval_batch_size,
            gather_batch_size=args.gather_batch_size,
            candidate_chunk_size=args.candidate_chunk_size,
            fusion_lambda=args.fusion_lambda,
            gpu_id=args.gpu_id,
        )


if __name__ == "__main__":
    main()
