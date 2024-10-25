import webdataset as wds
from pathlib import Path
import re
import io
from audiotools import AudioSignal
import dnsmos
from tqdm.auto import tqdm
import librosa
import torch
import argparse
import pandas as pd


def decode_audio(key, data):
    extension = re.sub(r".*[.]", "", key)
    if extension not in ["flac", "mp3", "sox", "wav", "m4a", "ogg", "wma"]:
        return None
    audio, sr = librosa.load(io.BytesIO(data), sr=None)
    signal = AudioSignal(audio, sample_rate=sr)
    if signal.sample_rate != 16000:
        signal = signal.resample(16000)
    if signal.num_channels > 1:
        signal = signal.to_mono()
    audio = signal.audio_data.squeeze(0).numpy()
    return audio, audio.shape[-1] / 16000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_idx", type=int, default=0)
    parser.add_argument("--num_dataloader_workers", type=int, default=8)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=".")
    
    args = parser.parse_args()

    with open("reazon_denoise_v2_urls.txt") as f:
        urls = [line.strip() for line in f]

    print(f"Total: {len(urls)}")
    urls = urls[args.worker_idx::args.num_workers]
    assert len(urls) >= args.num_dataloader_workers, f"Worker {args.worker_idx} has less than {args.num_dataloader_workers} urls"
    print(f"Worker: {args.worker_idx}, Total: {len(urls)}")
    hf_token = "<HF_TOKEN>"
    urls = [f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}'" for url in urls]

    ds = wds.WebDataset(urls).decode(decode_audio).to_tuple("__key__", "flac")
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=args.num_dataloader_workers, drop_last=False)

    compute = dnsmos.ComputeScore("sig_bak_ovr.onnx", "cuda", args.device_id)
    results = []
    score_dict = dict()
    scores = []
    for key, (audio, duration) in tqdm(dl):
        score = float(compute(audio, 16000, False)["OVRL"])
        results.append(
            {
                "key": key.replace("./", ""),
                "dnsmos": score,
                "duration": duration,
            }
        )
        score_dict[key] = (score, duration)
        scores.append(score)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_tsv_path = f"{args.output_dir}/reazon_denoise_v2_scores_{args.worker_idx}.tsv"
    df = pd.DataFrame(results)
    df.to_csv(output_tsv_path, sep="\t", index=False)
    print(f"Saved to {output_tsv_path}")
    # upload to huggingface
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=output_tsv_path,
        path_in_repo=f"reazon_denoise_v2_scores_{args.worker_idx}.tsv",
        repo_id="seastar105/denoised-reazonspeech-v2-dnsmos",
        repo_type="dataset",
    )