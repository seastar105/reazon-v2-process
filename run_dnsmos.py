import webdataset as wds
from pathlib import Path
import re
import io
from audiotools import AudioSignal
import dnsmos
import librosa
import torch
import pandas as pd
import ray
from huggingface_hub import HfApi


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


@ray.remote(num_cpus=4, num_gpus=0.5, max_retries=5)
def compute_scores(urls, output_dir):
    hf_token = "<HF_TOKEN>"
    urls = [f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}'" for url in urls]

    ds = wds.WebDataset(urls).decode(decode_audio).to_tuple("__key__", "flac")
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=4, drop_last=False)
    
    compute = dnsmos.ComputeScore("sig_bak_ovr.onnx", "cuda")
    results = []
    for key, (audio, duration) in dl:
        score = float(compute(audio, 16000, False)["OVRL"])
        results.append(
            {
                "key": key.replace("./", ""),
                "dnsmos": score,
                "duration": duration,
            }
        )

    names = [url.split("/")[-1].split(".")[0] for url in urls]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_name = "".join(names) + ".tsv"
    output_tsv_path = f"{output_dir}/{output_name}"
    df = pd.DataFrame(results)
    df.to_csv(output_tsv_path, sep="\t", index=False)
    # upload to huggingface
    
    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=output_tsv_path,
        path_in_repo=f"dnsmos/{output_name}",
        repo_id="seastar105/denoised-reazonspeech-v2-dnsmos",
        repo_type="dataset",
    )
    return df


if __name__ == "__main__":
    with open("reazon_denoise_v2_urls.txt", "r") as f:
        urls = [line.strip() for line in f]

    ray.init()
    chunk_size = 4
    url_chunks = [urls[i : i + chunk_size] for i in range(0, len(urls), chunk_size)]
    refs = [compute_scores.remote(urls, "./") for urls in url_chunks]
    df_list = ray.get(refs)
    df = pd.concat(df_list)
    df.to_csv("all_dnsmos.tsv", sep="\t", index=False)
    hf_token = "<HF_TOKEN>"
    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj="all_dnsmos.tsv",
        path_in_repo="all_dnsmos.tsv",
        repo_id="seastar105/denoised-reazonspeech-v2-dnsmos",
        repo_type="dataset",
    )
