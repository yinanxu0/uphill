from typing import Optional, Tuple, Union, Any, NamedTuple
import functools
import torchaudio
import numpy as np
import re
from io import BytesIO, IOBase
from packaging import version
from pathlib import Path
from subprocess import PIPE, run


from uphill.core.utils import (
    Pathlike, Channels, Seconds, Decibels,
    SmartOpen
)
from uphill.errors import AudioLoadingError
from .utils import compute_num_samples


def read_audio(
    path_or_fd: Union[Pathlike, Any],
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    # First handle special cases: OPUS and SPHERE (SPHERE may be encoded with shorten,
    #   which can only be decoded by binaries "shorten" and "sph2pipe").
    ## TODO: read bytes audio data
    try:
        if isinstance(path_or_fd, (str, Path)) and str(path_or_fd).lower().endswith(".opus"):
            return read_opus(
                path_or_fd,
                offset=offset,
                duration=duration,
            )
        return torchaudio_load(path_or_fd, offset=offset, duration=duration)
    except Exception as e:
        raise Exception(
            f"Reading audio from '{path_or_fd}' failed. Details: {type(e)}('{str(e)}')"
        )

def torchaudio_load(
    path: Union[Pathlike, Any], offset: float = 0.0, duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    # Need to grab the "info" about sampling rate before reading to compute
    # the number of samples provided in offset / num_frames.
    audio_info = torchaudio_info(path)
    frame_offset = 0
    num_frames = -1
    if offset > 0:
        frame_offset = compute_num_samples(offset, audio_info.samplerate)
    if duration is not None:
        num_frames = compute_num_samples(duration, audio_info.samplerate)
    if isinstance(path, IOBase):
        # Set seek pointer to the beginning of the file as torchaudio.info
        # might have left it at the end of the header
        path.seek(0)
    audio, sampling_rate = torchaudio.load(
        path,
        frame_offset=frame_offset,
        num_frames=num_frames,
    )
    return audio.numpy(), sampling_rate


def read_opus(
    path: Pathlike,
    offset: Seconds = 0.0,
    duration: Optional[Seconds] = None,
) -> Tuple[np.ndarray, int]:
    """
    Reads OPUS files either using torchaudio or ffmpeg.
    Torchaudio is faster, but if unavailable for some reason,
    we fallback to a slower ffmpeg-based implementation.

    :return: a tuple of audio samples and the sampling rate.
    """

    return read_opus_ffmpeg(
        path=path,
        offset=offset,
        duration=duration,
    )


def read_opus_ffmpeg(
    path: Pathlike,
    offset: Seconds = 0.0,
    duration: Optional[Seconds] = None,
    force_opus_sampling_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Reads OPUS files using ffmpeg in a shell subprocess.
    Unlike audioread, correctly supports offsets and durations for reading short chunks.
    Optionally, we can force ffmpeg to resample to the true sampling rate (if we know it up-front).

    :return: a tuple of audio samples and the sampling rate.
    """
    # Construct the ffmpeg command depending on the arguments passed.
    cmd = "ffmpeg -threads 1"
    sampling_rate = 48000
    # Note: we have to add offset and duration options (-ss and -t) BEFORE specifying the input
    #       (-i), otherwise ffmpeg will decode everything and trim afterwards...
    if offset > 0:
        cmd += f" -ss {offset}"
    if duration is not None:
        cmd += f" -t {duration}"
    # Add the input specifier after offset and duration.
    cmd += f" -i {path}"
    # Optionally resample the output.
    if force_opus_sampling_rate is not None:
        cmd += f" -ar {force_opus_sampling_rate}"
        sampling_rate = force_opus_sampling_rate
    # Read audio samples directly as float32.
    cmd += " -f f32le -threads 1 pipe:1"
    # Actual audio reading.
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    raw_audio = proc.stdout
    audio = np.frombuffer(raw_audio, dtype=np.float32)
    # Determine if the recording is mono or stereo and decode accordingly.
    try:
        channel_string = parse_channel_from_ffmpeg_output(proc.stderr)
        if channel_string == "stereo":
            new_audio = np.empty((2, audio.shape[0] // 2), dtype=np.float32)
            new_audio[0, :] = audio[::2]
            new_audio[1, :] = audio[1::2]
            audio = new_audio
        elif channel_string == "mono":
            audio = audio.reshape(1, -1)
        else:
            raise NotImplementedError(
                f"Unknown channel description from ffmpeg: {channel_string}"
            )
    except ValueError as e:
        raise AudioLoadingError(
            f"{e}\nThe ffmpeg command for which the program failed is: '{cmd}', error code: {proc.returncode}"
        )
    return audio, sampling_rate


def parse_channel_from_ffmpeg_output(ffmpeg_stderr: bytes) -> str:
    # ffmpeg will output line such as the following, amongst others:
    # "Stream #0:0: Audio: pcm_f32le, 16000 Hz, mono, flt, 512 kb/s"
    # but sometimes it can be "Stream #0:0(eng):", which we handle with regexp
    pattern = re.compile(r"^\s*Stream #0:0.*: Audio: pcm_f32le.+(mono|stereo).+\s*$")
    for line in ffmpeg_stderr.splitlines():
        try:
            line = line.decode()
        except UnicodeDecodeError:
            # Why can we get UnicodeDecoderError from ffmpeg output?
            # Because some files may contain the metadata, including a short description of the recording,
            # which may be encoded in arbitrarily encoding different than ASCII/UTF-8, such as latin-1,
            # and Python will not automatically recognize that.
            # We simply ignore these lines as they won't have any relevant information for us.
            continue
        match = pattern.match(line)
        if match is not None:
            return match.group(1)
    raise ValueError(
        f"Could not determine the number of channels for OPUS file from the following ffmpeg output "
        f"(shown as bytestring due to avoid possible encoding issues):\n{str(ffmpeg_stderr)}"
    )


class AudioInfo(NamedTuple):
    channels: int
    frames: int
    samplerate: int
    duration: float


@functools.lru_cache(maxsize=128, typed=False)
def torchaudio_info(path: Pathlike) -> AudioInfo:
    """
    Return an audio info data structure that's a compatible subset of ``pysoundfile.info()``
    that we need to create a ``Utterance`` manifest.
    """
    if (
        isinstance(path, (str, Path))
        and str(path).endswith(".mp3")
        and version.parse(torchaudio.__version__) >= version.parse("0.12.0")
    ):
        # Torchaudio 0.12 has a new StreamReader API that uses ffmpeg.
        # They dropped support for using sox bindings in torchaudio.info
        # for MP3 files and implicitly delegate the call to ffmpeg.
        # Unfortunately, they always return num_frames/num_samples = 0,
        # as explained here: https://github.com/pytorch/audio/issues/2524
        # We have to work around by streaming the MP3 and counting the number
        # of samples.

        streamer = torchaudio.io.StreamReader(src=str(path))
        assert streamer.num_src_streams == 1
        info = streamer.get_src_stream_info(0)
        streamer.add_basic_audio_stream(
            frames_per_chunk=int(info.sample_rate),
        )
        tot_samples = 0
        for (chunk,) in streamer.stream():
            tot_samples += chunk.shape[0]
        return AudioInfo(
            channels=info.num_channels,
            frames=tot_samples,
            samplerate=info.sample_rate,
            duration=tot_samples / info.sample_rate,
        )

    info = torchaudio.info(path)
    return AudioInfo(
        channels=info.num_channels,
        frames=info.num_frames,
        samplerate=info.sample_rate,
        duration=info.num_frames / info.sample_rate,
    )


@functools.lru_cache(maxsize=128, typed=False)
def urlaudio_info(url: Pathlike) -> AudioInfo:
    with SmartOpen.open(url, "rb") as f:
        source = BytesIO(f.read())
        samples, sampling_rate = read_audio(source)
    num_samples = (
        samples.shape[0] if len(samples.shape) == 1 else samples.shape[1]
    )
    duration = num_samples / sampling_rate
    return AudioInfo(
        channels=1,
        frames=num_samples,
        samplerate=sampling_rate,
        duration=duration
    )
    
