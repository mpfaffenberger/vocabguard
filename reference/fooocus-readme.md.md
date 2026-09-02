# Fooocus

Fooocus is an image generating software (based on Gradio).

Fooocus presents a rethinking of image generator designs. The software is offline, open source, and free, while at the same time, similar to many online image generators like Midjourney, the manual tweaking is not needed, and users only need to focus on the prompts and images. Fooocus has also simplified the installation: between pressing "download" and generating the first image, the number of needed mouse clicks is strictly limited to less than 3. Minimal GPU memory requirement is 4GB (Nvidia).

**Recently many fake websites exist on Google when you search “fooocus”. Do not trust those – here is the only official source of Fooocus.**

## Project Status: Limited Long-Term Support (LTS) with Bug Fixes Only

The Fooocus project, built entirely on the Stable Diffusion XL architecture, is now in a state of limited long-term support (LTS) with bug fixes only. As the existing functionalities are considered as nearly free of programmatic issues, future updates will focus exclusively on addressing any bugs that may arise.

There are no current plans to migrate to or incorporate newer model architectures. However, this may change during time with the development of open-source community.

For those interested in utilizing newer models such as Flux, we recommend exploring alternative platforms such as WebUI Forge, ComfyUI/SwarmUI. Additionally, several excellent forks of Fooocus are available for experimentation.

Again, recently many fake websites exist on Google when you search “fooocus”. Do NOT get Fooocus from those websites – this page is the only official source of Fooocus. We never have any website like such as “fooocus.com”, “fooocus.net”, “fooocus.co”, “fooocus.ai”, “fooocus.org”, “fooocus.pro”, “fooocus.one”. Those websites are ALL FAKE. They have ABSOLUTELY no relationship to us. Fooocus is a 100% non-commercial offline open-source software.

## Features

Fooocus enables high-quality text-to-image generation without needing much prompt engineering or parameter tuning. Fooocus has an offline GPT-2 based prompt processing engine and lots of sampling improvements so that results are always beautiful, no matter if your prompt is as short as “house in garden” or as long as 1000 words.

Fooocus supports image prompting, allowing you to use an input image to guide the generation process. Fooocus uses its own image prompt algorithm so that result quality and prompt understanding are more satisfying than other software that uses standard SDXL methods.

Fooocus provides advanced features for style, quality, aspect ratios, face swapping, and image description capabilities.

## Download

You can directly download Fooocus for Windows. After downloading, uncompress the file and run the "run.bat".

The first time you launch the software, it will automatically download models:
1. It will download default models to the folder "Fooocus\models\checkpoints" given different presets.
2. If you use inpaint, at the first time you inpaint an image, it will download Fooocus's own inpaint control model as the file "Fooocus\models\inpaint\inpaint_v26.fooocus.patch".

After Fooocus 2.1.60, you will also have run_anime.bat and run_realistic.bat. They are different model presets.

After Fooocus 2.3.0 you can also switch presets directly in the browser.

If you already have these files, you can copy them to the above locations to speed up installation.

## Minimal Requirement

Below is the minimal requirement for running Fooocus locally:
- Windows/Linux with Nvidia RTX 4XXX: 4GB GPU memory, 8GB system memory
- Windows/Linux with Nvidia RTX 3XXX: 4GB GPU memory, 8GB system memory  
- Windows/Linux with Nvidia RTX 2XXX: 4GB GPU memory, 8GB system memory
- Windows/Linux with Nvidia GTX 1XXX: 8GB GPU memory, 8GB system memory (only marginally faster than CPU)
- Windows/Linux with Nvidia GTX 9XX: 8GB GPU memory, 8GB system memory (faster or slower than CPU)
- Windows with AMD GPU: 8GB GPU memory, 8GB system memory (via DirectML)
- Linux with AMD GPU: 8GB GPU memory, 8GB system memory (via ROCm)
- Mac with M1/M2 MPS: Shared memory (about 9x slower than Nvidia RTX 3XXX)
- Windows/Linux/Mac CPU only: 0GB GPU memory, 32GB system memory (about 17x slower than Nvidia RTX 3XXX)

Note that Fooocus is only for extremely high quality image generating. We will not support smaller models to reduce the requirement and sacrifice result quality.

## UI Access and Authentication

In addition to running on localhost, Fooocus can also expose its UI in two ways:
* Local UI listener: use --listen (specify port e.g. with --port 8888).
* API access: use --share (registers an endpoint at .gradio.live).

In both ways the access is unauthenticated by default. You can add basic authentication by creating a file called auth.json in the main directory.

## Customization

After the first time you run Fooocus, a config file will be generated at Fooocus\config.txt. This file can be edited to change the model path or default parameters.

A safer way is just to try run_anime.bat or run_realistic.bat - they should already be good enough for different tasks.