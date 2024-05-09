import argparse
import asyncio
import json
import logging
import re
import yaml
import random
from typing import List

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm.asyncio import tqdm

from module.generate import tag_gen
from module.metainfo import SPECIAL, TARGET

logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_MODEL = "KBlueLeaf/DanTagGen-gamma"


class DanTagGenConfig:
    def __init__(self, model_paths: List[str], device: str):
        self.model_paths = model_paths
        self.device = device


class DanTagGen:
    def __init__(self, config: DanTagGenConfig):
        self.models = {
            model_path: [
                LlamaForCausalLM.from_pretrained(
                    model_path, attn_implementation="flash_attention_2"
                )
                .requires_grad_(False)
                .eval()
                .half()
                .to(config.device),
                LlamaTokenizer.from_pretrained(model_path),
            ]
            for model_path in config.model_paths
        }

    @torch.no_grad()
    async def generate_tags(
        self,
        model: str,
        rating: str = "",
        artist: str = "",
        characters: str = "",
        copyrights: str = "",
        target: str = "long",
        special_tags: List[str] = ["1girl"],
        general: str = "",
        aspect_ratio: float = 0.0,
        blacklist: str = "",
        escape_bracket: bool = False,
        temperature: float = 1.35,
    ) -> str:
        try:
            llama_model, llama_tokenizer = self.models[model]
            prompt = self._create_prompt(
                rating,
                artist,
                characters,
                copyrights,
                target,
                special_tags,
                general,
                aspect_ratio,
            )
            black_list_patterns = [
                re.compile(pattern.strip()) for pattern in blacklist.split(",")
            ]
            prompt_tags = special_tags + general.strip().strip(",").split(",")
            len_target = TARGET[target]

            llm_gen, extra_tokens = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: next(
                    tag_gen(
                        llama_model,
                        llama_tokenizer,
                        prompt,
                        prompt_tags,
                        len_target,
                        black_list_patterns,
                        temperature=temperature,
                        top_p=0.95,
                        top_k=100,
                        max_new_tokens=256,
                        max_retry=5,
                    )
                ),
            )

            general = f"{general.strip().strip(',')}, {','.join(extra_tokens)}"
            tags = [tag.strip() for tag in general.strip().split(",") if tag.strip()]
            filtered_tags = [
                tag
                for tag in tags
                if not any(pattern.match(tag) for pattern in black_list_patterns)
            ]

            special = special_tags + [tag for tag in filtered_tags if tag in SPECIAL]
            tags = [tag for tag in filtered_tags if tag not in special]

            final_prompt = self._create_final_prompt(
                special, characters, copyrights, artist, tags, rating, escape_bracket
            )
            return final_prompt.replace("\n", " ")
        except Exception as e:
            logging.error(f"Error generating tags: {str(e)}")
            raise

    def _create_prompt(
        self,
        rating: str,
        artist: str,
        characters: str,
        copyrights: str,
        target: str,
        special_tags: List[str],
        general: str,
        aspect_ratio: float,
    ) -> str:
        artist = artist.strip() or "<|empty|>"
        characters = characters.strip() or "<|empty|>"
        copyrights = copyrights.strip() or "<|empty|>"
        general = general.strip().strip(",")
        return f"""
rating: {rating or '<|empty|>'}
artist: {artist}
characters: {characters}
copyrights: {copyrights}
aspect ratio: {f"{aspect_ratio:.1f}" or '<|empty|>'}
target: {'<|' + target + '|>' if target else '<|long|>'}
general: {", ".join(special_tags)}, {general}<|input_end|>
""".strip()

    def _create_final_prompt(
        self,
        special: List[str],
        characters: str,
        copyrights: str,
        artist: str,
        tags: List[str],
        rating: str,
        escape_bracket: bool,
    ) -> str:
        prompt_components = [", ".join(special)]
        if characters:
            prompt_components.append(characters)
        if copyrights:
            prompt_components.append(copyrights)
        if artist:
            prompt_components.append(artist)
        prompt_components.append(", ".join(tags))
        prompt_components.append(rating)

        final_prompt = ", ".join(prompt_components)

        if escape_bracket:
            final_prompt = final_prompt.translate(
                str.maketrans({"[": "\\[", "]": "\\]", "(": "\\(", ")": "\\)"})
            )
        return final_prompt


async def batch_inference(
    tag_generator: DanTagGen,
    input_data: dict,
    num_prompts: int,
    output_file: str,
    aspect_ratios: List[str],
    debug: bool = False,
) -> None:
    if debug:
        logging.info(f"Starting batch inference for {num_prompts} prompts")
    results = []

    async def generate_prompt(i):
        try:
            aspect_ratio = random.choice(aspect_ratios)  # Select a random aspect ratio
            width, height = aspect_ratio.split(" x ")
            input_data["aspect_ratio"] = float(width) / float(height)
            prompt = await tag_generator.generate_tags(**input_data)
            result = {"prompt": prompt, "resolution": aspect_ratio}
            if debug:
                logging.info("")
                logging.info(f"\tPrompt: {prompt}")
                logging.info(f"\tResolution: {aspect_ratio}")
            return result
        except Exception as e:
            logging.error(f"Error generating prompt: {str(e)}")
            return {"prompt": "", "resolution": aspect_ratio}

    results = await tqdm.gather(*[generate_prompt(i) for i in range(num_prompts)])

    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)
    logging.info(f"Batch inference completed. Prompts saved to: {output_file}")


async def main():
    parser = argparse.ArgumentParser(description="Danbooru Tags-based Prompt Generator")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Name or path of the pre-trained model",
    )
    parser.add_argument(
        "--rating", type=str, default="general", help="Rating of the generated prompt"
    )
    parser.add_argument(
        "--artist", type=str, default="", help="Artist to include in the prompt"
    )
    parser.add_argument(
        "--characters", type=str, default="", help="Characters to include in the prompt"
    )
    parser.add_argument(
        "--copyrights", type=str, default="", help="Copyrights to include in the prompt"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="short",
        help="Target length of the generated prompt",
    )
    parser.add_argument(
        "--special_tags",
        type=str,
        default="1girl,solo",
        help="Special tags to include in the prompt (comma-separated)",
    )
    parser.add_argument(
        "--general",
        type=str,
        default="",
        help="Additional general tags to include in the prompt (comma-separated)",
    )
    parser.add_argument(
        "--blacklist",
        type=str,
        default=".*background, .*text.*, .*blurry.*, comic, .*manga.*, .*magazine.*, .*username.*, artist name",
        help="Tags to exclude from the generated prompt (comma-separated)",
    )
    parser.add_argument(
        "--escape_bracket",
        action="store_true",
        help="Whether to escape brackets in the generated prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.35,
        help="Temperature value for text generation",
    )
    parser.add_argument(
        "--num_prompts", type=int, default=10, help="Number of prompts to generate"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="prompt.json",
        help="Output file path to save the generated prompts",
    )
    parser.add_argument(
        "--aspect_ratios",
        type=str,
        default="1024 x 1024,1152 x 896,896 x 1152,1216 x 832,832 x 1216,1344 x 768,768 x 1344,1536 x 640,640 x 1536",
        help="Aspect ratios for the generated prompts (comma-separated, in 'width x height' format)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the generated prompts and resolutions for debugging",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, "r") as file:
            config_data = yaml.safe_load(file)
        for key, value in config_data.get("config", {}).items():
            if hasattr(args, key):
                if isinstance(value, list):
                    if key == "blacklist":
                        setattr(args, key, ",".join(str(item) for item in value))
                    else:
                        setattr(args, key, value)
                else:
                    setattr(args, key, value)
        if args.debug:
            print("YAML Configuration:")
            print(yaml.dump(config_data, default_flow_style=False))
    input_data = {
        "model": args.model,
        "rating": args.rating,
        "artist": args.artist,
        "characters": args.characters,
        "copyrights": args.copyrights,
        "target": args.target,
        "special_tags": args.special_tags,
        "general": args.general,
        "blacklist": args.blacklist,
        "escape_bracket": args.escape_bracket,
        "temperature": args.temperature,
    }

    config = DanTagGenConfig(
        model_paths=[input_data["model"]],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    tag_generator = DanTagGen(config)

    if args.debug:
        logging.info(f"Starting batch inference with input data: {input_data}")
    await batch_inference(
        tag_generator,
        input_data,
        args.num_prompts,
        args.output_file,
        args.aspect_ratios,
        args.debug,
    )


if __name__ == "__main__":
    asyncio.run(main())
