import argparse
import json
import os
import pickle
import time
import warnings
from pathlib import Path

import yaml

import datasets
import open_clip
import torch
from clip_benchmark.datasets.builder import image_captions_collate_fn
from clip_benchmark.metrics import zeroshot_retrieval as zsr

warnings.filterwarnings("ignore", message="Length of IterableDataset")

#Taken from eval_utils.wds_eval.py
def create_model(model_arch, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model_path = str(model_path)
    model, _, transform = open_clip.create_model_and_transforms(
        model_arch, pretrained=model_path
    )
    model.eval()
    # model.half()
    model = model.to(device)

    return model, transform, device

#Taken from eval_utils.retr_eval.py
class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        super().__init__()
        self._dataset = hf_dataset
        self.transform = (lambda x: x) if transform is None else transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int):
        return (
            self.transform(self._dataset[index]["image"]),
            self._dataset[index]["caption"],
        )

def evaluate_retrieval_dataset(
    task, model_arch, model_path, data_root=None, batch_size=64, num_workers=4
):
    """Evaluate CLIP model on retrieval task."""

    model, transform, device = create_model(model_arch, model_path)
    tokenizer = open_clip.get_tokenizer(model_arch)

    dataset = RetrievalDataset(
        datasets.load_dataset(
            f"nlphuji/{task.replace('retrieval/', '')}",
            split="test",
            cache_dir=os.path.join(data_root, "hf_cache")
            if data_root is not None
            else None,
        ),
        transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=image_captions_collate_fn,
    )

    metrics = zsr.evaluate(
        model, dataloader, tokenizer, recall_k_list=[1, 5, 10], device=device
    )
    metrics["mean_recall@1"] = 0.5 * (
        metrics["text_retrieval_recall@1"] + metrics["image_retrieval_recall@1"]
    )
    return metrics

def evaluate_model(task_key, train_info, data_root, dataset_size, batch_size=64):
    if task_key.startswith("retrieval/"):
        metrics = evaluate_retrieval_dataset(
            task_key,
            train_info["scale_config"]["model"],
            train_info["checkpoint"],
            data_root=data_root,
            batch_size=batch_size,
        )
    else:
        metrics = None
        
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Path to output directory to use for evaluation. If nothing is passed, use the training output dir.",
    )
    parser.add_argument(
        "--data_dir",
        help="(Optional) Path to directory containing downloaded evaluation datasets.",
        default=None,
    )
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")

    # Debug-only flags. Using any of these might invalidate your submission.
    parser.add_argument(
        "--use_model",
        type=str,
        help='If set, manually specify a model architecture and checkpoint path ("model path")',
        default=None,
    )

    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)

    if args.use_model is not None:
        args.train_output_dir = args.output_dir
        # Generate barebones info.pkl
        model_arch, model_checkpoint = args.use_model.split(maxsplit=1)
        Path.mkdir(args.output_dir, parents=True, exist_ok=True)
        with open(args.train_output_dir / "info.pkl", "wb") as f:
            pickle.dump(
                {"scale_config": {"model": model_arch}, "checkpoint": model_checkpoint},
                f,
            )

    # Read training information
    train_info_filename = args.train_output_dir / "info.pkl"
    train_info = pickle.load(open(train_info_filename, "rb"))

    results_filename = args.output_dir / "eval_results.jsonl"

    # Get list of datasets
    with open(os.path.join(os.path.dirname(__file__), "tasklist.yml")) as f:
        tasks = yaml.safe_load(f)

    # Check for cached results
    results = {}
    cached_train_info_filename = args.output_dir / "info.pkl"
    if args.output_dir.exists() and cached_train_info_filename.exists():
        # If the output directory already exists, the training information should match.
        cached_train_info = pickle.load(open(cached_train_info_filename, "rb"))
        error_message = (
            "Error: output directory exists, but the training configs do not match. "
            "If you are re-using an output directory for evals, please be sure that "
            "the training output directory is consistent."
        )
        assert cached_train_info == train_info, error_message

        # Read existing results
        if results_filename.exists():
            with open(results_filename, "r") as f:
                lines = [json.loads(s) for s in f.readlines()]
                for line in lines:
                    if line["key"] not in tasks:
                        continue
                    results[line["dataset"]] = line
            print(f"Found {len(results)} eval result(s) in {results_filename}.")
    else:
        Path.mkdir(args.output_dir, parents=True, exist_ok=True)
        pickle.dump(train_info, open(cached_train_info_filename, "wb"))

    train_checkpoint = Path(train_info["checkpoint"])
    try:
        exists = Path(train_info["checkpoint"]).exists()
    except:
        exists = False
    if not exists and args.use_model is None:
        print(
            "Warning, did not find or could not read checkpoint at",
            train_info["checkpoint"],
        )
        default_checkpoint_name = (
            args.train_output_dir / "checkpoints" / "epoch_latest.pt"
        )
        print("Defaulting to", default_checkpoint_name)
        train_info["checkpoint"] = default_checkpoint_name

    print("Evaluating")

    starttime = int(time.time())

    for task_key in tasks:
        task_name = tasks[task_key].get("name", task_key)
        if task_name in results:
            print(
                f"Skipping {task_name} since results are already in {results_filename}"
            )
        else:
            print(f"Evaluating on {task_name}")
            metrics = evaluate_model(
                task_key,
                train_info,
                args.data_dir,
                tasks[task_key].get("size"),
                batch_size=args.batch_size,
            )
            metrics["main_metric"] = metrics.get(
                tasks[task_key].get("main_metric", "acc1")
            )
            results[task_name] = {
                "key": task_key,
                "dataset": task_name,
                "metrics": metrics,
            }
            with open(results_filename, "a+") as f:
                f.write(json.dumps(results[task_name]) + "\n")

        if results[task_name]["metrics"]["main_metric"] is not None:
            print(f"Score: {results[task_name]['metrics']['main_metric']:.4f}")
        else:
            print(f"Score: No summary metric")

    elapsed = int(time.time()) - starttime
    print(
        f"Evaluation time: {elapsed // 3600} hour(s) {elapsed % 3600 // 60} minute(s) {elapsed % 60} second(s)"
    )
    print()
    print("=== Final results ===")
    for line in results.values():
        print(f"{line['dataset']}: {line['metrics']['main_metric']}")

