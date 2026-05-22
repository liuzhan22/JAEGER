# ./utils/audio_process.py

import base64
from io import BytesIO

import audioread
import av
import librosa
import numpy as np


SAMPLE_RATE = 16000
def _check_if_video_has_audio(video_path):
    container = av.open(video_path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return False
    return True


def process_audio_info(conversations: list[dict] | list[list[dict]], use_audio_in_video: bool):
    """
    Read and process audio info

    Support dict keys:

    type = audio
    - audio
    - audio_start
    - audio_end

    type = video
    - video
    - video_start
    - video_end
    """
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" in ele or "audio_url" in ele:
                        path = ele.get("audio", ele.get("audio_url"))
                        audio_start = ele.get("audio_start", 0.0)
                        audio_end = ele.get("audio_end", None)
                        # numpy input: now support both mono [T] and multi-channel [C, T] or [T, C]
                        if isinstance(path, np.ndarray):
                            audio_np = path
                            # Normalize shape to [T] or [C, T]
                            if audio_np.ndim == 1:
                                # mono, keep behavior
                                sliced = audio_np[
                                    int(SAMPLE_RATE * audio_start) : None
                                    if audio_end is None
                                    else int(SAMPLE_RATE * audio_end)
                                ]
                                audios.append(sliced)
                                continue

                            if audio_np.ndim == 2:
                                # accept [T, C] or [C, T]; convert to [C, T]
                                if audio_np.shape[0] < audio_np.shape[1]:
                                    # heuristic: treat first dim as time
                                    audio_np = audio_np.T

                                # slice along time axis
                                t_start = int(SAMPLE_RATE * audio_start)
                                t_end = (
                                    None
                                    if audio_end is None
                                    else int(SAMPLE_RATE * audio_end)
                                )
                                audio_np = audio_np[:, t_start:t_end]
                                audios.append(audio_np)
                                continue

                            # For higher-dim inputs, fall back to original error
                            raise ValueError(
                                f"Unsupported audio numpy shape {audio_np.shape}, expected 1D or 2D"
                            )
                        elif path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            data = BytesIO(base64.b64decode(base64_data))
                        elif path.startswith("http://") or path.startswith("https://"):
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
                elif use_audio_in_video and ele["type"] == "video":
                    if "video" in ele or "video_url" in ele:
                        path = ele.get("video", ele.get("video_url"))
                        audio_start = ele.get("video_start", 0.0)
                        audio_end = ele.get("video_end", None)
                        assert _check_if_video_has_audio(
                            path
                        ), "Video must has audio track when use_audio_in_video=True"
                        if path.startswith("http://") or path.startswith("https://"):
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown video {}".format(ele))
                else:
                    continue
                # librosa.load returns mono by default; for FOA / multichannel, we keep stereo/FOA
                # by setting mono=False, so downstream can see full channel information.
                y, _ = librosa.load(
                    data,
                    sr=SAMPLE_RATE,
                    offset=audio_start,
                    duration=(audio_end - audio_start) if audio_end is not None else None,
                    mono=False,
                )
                audios.append(y)
    if len(audios) == 0:
        audios = None
    return audios
