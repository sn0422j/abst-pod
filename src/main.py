import argparse
import configparser
import dataclasses
import os
import shutil
from pathlib import Path

import ffmpeg
import pandas as pd
import soundfile as sf
import torch
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
from pybars import Compiler
from tqdm import tqdm

# cspell: words tqdm pybars espnet soundfile minlenratio maxlenratio tacotron vits iterrows


@dataclasses.dataclass(kw_only=True)
class Config:
    # espnet settings
    model_tag: str
    vocoder_tag: str
    device: str
    speed_control_alpha: str
    speed_control_alpha_float: float = dataclasses.field(init=False)

    # load & save settings
    csv_load_dir: str
    csv_file_name: str
    audio_save_dir: str
    make_zip_flag: str
    make_zip_flag_bool: bool = dataclasses.field(init=False)

    # template settings
    input_text_template_path: str
    output_file_name_template_path: str
    output_lrc_text_template_path: str

    def __post_init__(self):
        self.speed_control_alpha_float = float(self.speed_control_alpha)
        self.make_zip_flag_bool = self.make_zip_flag.lower() == "true"


def main(config: Config):
    TEXT2SPEECH_THRESHOLD = 0.5
    TEXT2SPEECH_MINLENRATIO = 0.0
    TEXT2SPEECH_MAXLENRATIO = 10.0
    TEXT2SPEECH_BACKWARD_WINDOW = 1
    TEXT2SPEECH_FORWARD_WINDOW = 3
    TEXT2SPEECH_USE_ATT_CONSTRAINT = False
    TEXT2SPEECH_NOISE_SCALE = 0.333
    TEXT2SPEECH_NOISE_SCALE_DUR = 0.333

    # preparing text2speech
    text2speech = Text2Speech.from_pretrained(
        model_tag=str_or_none(config.model_tag),
        vocoder_tag=str_or_none(config.vocoder_tag),
        device=str_or_none(config.device),
        # Only for Tacotron 2 & Transformer
        threshold=TEXT2SPEECH_THRESHOLD,
        # Only for Tacotron 2
        minlenratio=TEXT2SPEECH_MINLENRATIO,
        maxlenratio=TEXT2SPEECH_MAXLENRATIO,
        backward_window=TEXT2SPEECH_BACKWARD_WINDOW,
        forward_window=TEXT2SPEECH_FORWARD_WINDOW,
        use_att_constraint=TEXT2SPEECH_USE_ATT_CONSTRAINT,
        # Only for FastSpeech & FastSpeech2 & VITS
        speed_control_alpha=config.speed_control_alpha_float,
        # Only for VITS
        noise_scale=TEXT2SPEECH_NOISE_SCALE,
        noise_scale_dur=TEXT2SPEECH_NOISE_SCALE_DUR,
    )

    audio_save_dir = Path("audio").joinpath(config.audio_save_dir)
    os.makedirs(audio_save_dir, exist_ok=True)

    # load template
    compiler = Compiler()
    with open(Path(config.input_text_template_path), "r") as f:
        input_text_template = compiler.compile(f.read())
    with open(Path(config.output_file_name_template_path), "r") as f:
        output_file_name_template = compiler.compile(f.read().strip())
    with open(Path(config.output_lrc_text_template_path), "r") as f:
        output_lrc_text_template = compiler.compile(f.read().strip())

    # load csv
    df = pd.read_csv(Path(config.csv_load_dir).joinpath(config.csv_file_name))

    # run loop
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            index_str = str(i).zfill(len(str(len(df))))
            input_text = input_text_template(row.to_dict() | {"index": index_str})
            assert isinstance(input_text, str)

            # generate audio
            with torch.no_grad():
                wav = text2speech(input_text)["wav"]

            # save audio
            sf.write(
                Path("audio").joinpath("out.wav"),
                wav.view(-1).cpu().numpy(),
                text2speech.fs,
                "PCM_16",
            )

            # convert wav to mp3
            file_name = output_file_name_template(row.to_dict() | {"index": index_str})
            assert isinstance(file_name, str)
            ffmpeg.run(
                ffmpeg.output(
                    ffmpeg.input(str(Path("audio").joinpath("out.wav"))),
                    f"{str(audio_save_dir.joinpath(file_name))}.mp3",
                ),
                overwrite_output=True,
            )

            # generate lrc file
            with open(f"{str(audio_save_dir.joinpath(file_name))}.lrc", "w") as f:
                f.write(str(output_lrc_text_template(row.to_dict() | {"index": index_str})).replace("\n", ""))

        except Exception as error:
            print(error)
            continue

    if config.make_zip_flag_bool:
        # make zip archive
        shutil.make_archive(
            str(audio_save_dir),
            format="zip",
            root_dir=str(audio_save_dir),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.ini", type=str)
    args = parser.parse_args()
    assert isinstance(args.config, str)

    config_reader = configparser.RawConfigParser()
    config_reader.read(args.config)
    config = Config(**config_reader["settings"])  # type: ignore
    print(f"{config=}")

    main(config)
