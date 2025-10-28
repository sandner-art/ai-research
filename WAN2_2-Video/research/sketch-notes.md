# Wan2.2 GGUF: Images and Sound to Video - Complete Guide

Daniel Sandner, research sketch for article on https://sandner.art/

## Introduction

Wan2.2 represents a major leap in open-source video generation, bringing cinematic-quality AI video creation to consumer hardware. Through GGUF quantization and ComfyUI integration, this powerful 14B parameter model becomes accessible even on GPUs with 16GB VRAM like the NVIDIA RTX A4000.

This guide covers running Wan2.2's Speech-to-Video (S2V), Image-to-Video (I2V), Text-to-Video (T2V), and the hybrid Text+Image-to-Video (TI2V) models locally using GGUF quantized versions in ComfyUI.

## What is Wan2.2?

Wan2.2 is an advanced video generation model developed by Alibaba Cloud's Wan-AI team. It introduces several groundbreaking features:

### Key Innovations

**Mixture-of-Experts (MoE) Architecture**: The 14B models use a dual-expert design with high-noise and low-noise specialists. Each expert handles 14B parameters (27B total), but only 14B are active per inference step, maintaining efficiency while increasing model capacity.

**Cinematic Aesthetics**: Trained on meticulously curated data with detailed labels for lighting, composition, contrast, and color tone, enabling precise control over visual style.

**Complex Motion Generation**: Enhanced training data (+83.2% more videos than Wan2.1) enables smooth reproduction of complex movements with improved naturalness and controllability.

**Multiple Model Variants**:
- **T2V-A14B**: Text-to-Video at 480P/720P
- **I2V-A14B**: Image-to-Video at 480P/720P  
- **TI2V-5B**: Hybrid Text/Image-to-Video at 720P (efficient 5B model)
- **S2V-14B**: Speech/Audio-to-Video with lip-sync capabilities

## Why GGUF for Wan2.2?

GGUF (GPT-Generated Unified Format) quantization dramatically reduces VRAM requirements while maintaining acceptable quality:

### GGUF Quantization Levels

| Quantization | Model Size (14B) | Model Size (5B) | Quality | Speed | Recommended For |
|--------------|------------------|-----------------|---------|-------|-----------------|
| Q2_K | 5.3 GB | 1.85 GB | Lowest | Fastest | Testing only |
| Q3_K_S | 6.51 GB | 2.29 GB | Low | Very Fast | Quick drafts |
| Q3_K_M | 7.17 GB | 2.55 GB | Low-Med | Very Fast | Budget GPUs |
| Q4_K_S | 8.75 GB | 3.12 GB | Medium | Fast | 8-12GB VRAM |
| Q4_K_M | 9.65 GB | 3.43 GB | Medium | Fast | **RTX A4000 Sweet Spot** |
| Q5_K_S | 10.1 GB | 3.56 GB | Good | Moderate | 12-16GB VRAM |
| Q5_K_M | 10.8 GB | 3.81 GB | Good | Moderate | Quality balance |
| Q6_K | 12 GB | 4.21 GB | Very Good | Slower | Quality priority |
| Q8_0 | 15.4 GB | 5.4 GB | Excellent | Slow | Near FP16 quality |

### RTX A4000 Recommendations

For the 16GB RTX A4000:
- **Best Balance**: Q5_K_M or Q6_K (leaves VRAM for other components)
- **Maximum Quality**: Q8_0 (tight but usable)
- **Faster Generation**: Q4_K_M with Lightning LoRA

## System Requirements

### Minimum Setup
- **GPU**: 12GB VRAM (Q4 quantization)
- **RAM**: 32GB system RAM recommended
- **Storage**: 50GB+ free space
- **OS**: Windows 10/11, Linux

### Optimal Setup (RTX A4000)
- **GPU**: 16GB VRAM
- **RAM**: 64GB system RAM
- **Storage**: SSD with 100GB+ free
- **CUDA**: 12.4+ for FP8 support

## Installation Guide

### Step 1: Update ComfyUI

Ensure you have the latest ComfyUI version:

**Via ComfyUI Interface**:
- Click "Manager" → "Update ComfyUI"

**Windows Portable**:
```bash
cd ComfyUI_windows_portable\update
update_comfyui.bat
```

**Manual Installation**:
```bash
cd ComfyUI
git pull origin master
pip install -r requirements.txt
```

### Step 2: Install ComfyUI-GGUF Node

Install city96's GGUF custom node:

1. Open ComfyUI Manager
2. Click "Custom Nodes Manager"
3. Search for "ComfyUI-GGUF" by city96
4. Click Install (or Update if already installed)
5. Restart ComfyUI

### Step 3: Download Required Models

Create the following folder structure if it doesn't exist:

```
ComfyUI/
├── models/
│   ├── unet/              # GGUF models go here
│   ├── vae/               # VAE models
│   ├── text_encoders/     # Text encoders
│   ├── audio_encoders/    # For S2V models
│   └── loras/             # Lightning LoRA files
```

### Core Models Required

**1. VAE** (all models):
- File: `wan_2.1_vae.safetensors` or `wan2.2_vae.safetensors`
- Location: `ComfyUI/models/vae/`
- Source: [Kijai WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/tree/main)

**2. Text Encoder** (all models):
- File: `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- Location: `ComfyUI/models/text_encoders/`
- Source: [Comfy-Org](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)

**3. Audio Encoder** (S2V only):
- File: `wav2vec2_large_english_fp16` (folder)
- Location: `ComfyUI/models/audio_encoders/`
- Source: [Comfy-Org](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)

### Model-Specific GGUF Files

**Speech-to-Video (S2V-14B)**:
- Source: [QuantStack/Wan2.2-S2V-14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-S2V-14B-GGUF)
- Location: `ComfyUI/models/unet/`
- Recommended: Q5_K_M (15 GB) or Q4_K_M (13.9 GB)

**Image-to-Video (I2V-A14B)**:
- Source: [QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF)
- High Noise: Download matching pair (e.g., Q5_K_M high & low)
- Low Noise: Must match the same quantization level
- Location: `ComfyUI/models/unet/`

**Text-to-Video (T2V-A14B)**:
- Source: [QuantStack/Wan2.2-T2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF)
- High Noise + Low Noise pairs required
- Location: `ComfyUI/models/unet/`

**Text+Image-to-Video (TI2V-5B)**:
- Source: [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF)
- Single model file (no high/low noise split)
- Recommended: Q6_K (4.21 GB) for best quality on A4000
- Location: `ComfyUI/models/unet/`

### Optional: Lightning LoRA (Speed Boost)

Lightning LoRA reduces steps from 20 to 4 with minimal quality loss:

**Download**:
- [Wan2.2 Lightning LoRA](https://huggingface.co/Thelocallab/WAN-2.1-loras)
- Location: `ComfyUI/models/loras/`

**Settings with Lightning LoRA**:
- Steps: 4
- CFG Scale: 1.0
- Sampler: euler or res_multistep

## Workflow Setup

### Loading Workflows

ComfyUI includes native Wan2.2 workflows:

1. Open ComfyUI
2. Go to **Workflow → Browse Templates → Video**
3. Select:
   - "Wan2.2 14B S2V" for Speech-to-Video
   - "Wan2.2 14B I2V" for Image-to-Video
   - "Wan2.2 14B T2V" for Text-to-Video
   - "Wan2.2 5B Ti2V" for Text+Image-to-Video

### Key Nodes Explained

**UNet Loader (GGUF)**:
- Replaces standard "Load Diffusion Model" node
- Select your GGUF quantization level
- For I2V/T2V: Load both high and low noise models

**Load VAE**:
- Always use `wan_2.1_vae.safetensors` or `wan2.2_vae.safetensors`

**CLIP Text Encoder**:
- Load `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- Supports multi-language prompts

**Load Audio** (S2V only):
- Accepts WAV, MP3 formats
- Match audio length to desired video length

## Optimal Settings for RTX A4000

### General Configuration

**Resolution Settings**:
- **480P**: 832×480 or 480×832 (portrait)
- **720P**: 1280×720 or 720×1280 (portrait)
- **Aspect Ratios**: Maintain standard ratios (16:9, 9:16, 4:3)

**Frame Settings**:
- FPS: 24 (standard)
- Frame Count: 
  - 2 seconds = 49 frames (24 fps × 2 + 1)
  - 3 seconds = 73 frames
  - 5 seconds = 121 frames

**Sampling Parameters**:

*Without Lightning LoRA*:
- Steps: 20-30
- CFG Scale: 6.0-7.5
- Sampler: euler or dpmpp_2m
- Scheduler: normal

*With Lightning LoRA*:
- Steps: 4
- CFG Scale: 1.0
- Sampler: euler
- Scheduler: simple

### Memory Optimization Tips

1. **Enable Model Offloading**: 
   - In ComfyUI settings, enable "Automatically unload models"
   - Loads models to RAM when not in use

2. **Lower Resolution First**:
   - Test at 480P before attempting 720P
   - Upscale in post-processing if needed

3. **Batch Size**:
   - Keep at 1 for GGUF models
   - Use frame extension for longer videos

4. **Quantization Choice**:
   - Q4_K_M: ~900 seconds for 81 frames at 840×420
   - Q5_K_M: Better quality, 20% slower
   - Q6_K: Near-FP16 quality, 30-40% slower

### Generation Time Estimates (RTX A4000)

| Resolution | Frames | Quantization | Lightning LoRA | Time (approx) |
|------------|--------|--------------|----------------|---------------|
| 480P | 49 (2s) | Q4_K_M | Yes | 4-6 min |
| 480P | 49 (2s) | Q5_K_M | No | 10-12 min |
| 720P | 73 (3s) | Q5_K_M | Yes | 6-9 min |
| 720P | 121 (5s) | Q5_K_M | No | 18-22 min |
| 1024×574 | 121 (5s) | Q8_0 | No | 7 min* |

*With Sage Attention 2.2 enabled

## Advanced Optimization: Sage Attention

Sage Attention can provide 30-50% speed improvements without quality loss.

### Installation (Windows)

**Prerequisites**:
- CUDA Toolkit 12.4+
- Visual Studio Build Tools
- Python 3.10-3.12

**Install Triton**:
```bash
cd ComfyUI
python_embeded\python.exe -m pip install triton-windows
```

**Install Sage Attention**:
```bash
python_embeded\python.exe -m pip install sageattention
```

### Usage

**Method 1: Launch Parameter**:
```bash
python main.py --use-sage-attention
```

**Method 2: Custom Node**:
- Install "ComfyUI-SageAttention" via Manager
- Add SageAttention node to workflow

**Note**: Sage Attention 2.2 works best with RTX 30xx/40xx cards. RTX 50xx supports Sage Attention 3 for even greater speedups.

## Prompt Engineering for Wan2.2

### Effective Prompting Structure

Wan2.2 benefits from detailed, structured prompts:

```
[Scene Description], [Subject Details], [Motion Description], 
[Camera Work], [Lighting], [Style], [Color Grading]
```

### Examples

**Good Text-to-Video Prompt**:
```
A serene mountain lake at golden hour, crystal clear water 
reflecting the sky. A lone canoe glides smoothly across the 
water from left to right. Cinematic wide shot, soft natural 
lighting, warm color tones, subtle depth of field. Professional 
cinematography, 24fps.
```

**Good Image-to-Video Prompt**:
```
The cat slowly turns its head toward the camera, ears perking 
up with curiosity. Natural lighting from window, subtle 
breathing motion, whiskers twitching. Maintain photorealistic 
style, shallow depth of field.
```

**Good Speech-to-Video Prompt**:
```
Professional headshot style, person speaking naturally to 
camera. Subtle head movements, natural eye contact, relaxed 
posture. Studio lighting with soft shadows, clean background, 
cinematic color grading.
```

### Prompt Enhancement Options

**Built-in Enhancement** (official Wan2.2):
- Uses Qwen2.5 models locally
- Supports Dashscope API for cloud enhancement

**ComfyUI Solutions**:

1. **LLM Prompt Enhancer Node**:
   - GitHub: [comfyui-llm-prompt-enhancer](https://github.com/pinkpixel-dev/comfyui-llm-prompt-enhancer)
   - Supports local Ollama models
   - Recommended: qwen2.5:1.5b, gemma2:2b
   - 50+ artistic styles

2. **Qwen-Image Prompt Enhancement**:
   - Uses Qwen2.5-VL models for vision-language prompting
   - Better for I2V workflows
   - Requires additional VRAM (7B model)

### Local Prompt Enhancement Workflow

**Using Ollama + LLM Prompt Enhancer**:

1. Install Ollama: https://ollama.ai
2. Download model: `ollama pull qwen2.5:1.5b`
3. Install LLM Prompt Enhancer in ComfyUI
4. Configure:
   - Provider: ollama
   - Model: qwen2.5:1.5b
   - Host: http://localhost:11434
5. Connect to CLIP encoder in workflow

**Benefits**:
- Free, runs locally
- No API costs
- Works offline
- Low VRAM overhead

## Model-Specific Guides

### Speech-to-Video (S2V) Workflow

**Unique Features**:
- Lip-sync to audio
- Natural facial movements
- Character animation from single image

**Key Settings**:
- Match video length to audio duration
- Use batch sizes for frame chunks
- Audio Crop node for timing control

**Workflow Tips**:
1. Load reference image (portrait, facing camera)
2. Load audio file (clean speech)
3. Set resolution (1024×704 recommended)
4. Calculate frames: `(audio_seconds × 24) + 1`
5. Generate with appropriate CFG (6.0-6.6)

**Performance**:
- Q4 GGUF: ~5 seconds generation for 5s video
- BF16: ~11 seconds for same output
- Quality difference minimal at Q5+

### Image-to-Video (I2V) Workflow

**Best Practices**:
- Use high-quality source images
- Aspect ratio maintained automatically
- Describe motion clearly in prompt

**Common Issues**:
- First frame "overbaked": Use first-frame fix nodes
- Limited motion: Add motion keywords (flowing, moving, drifting)
- Style shift: Specify "maintain original style" in prompt

### Text-to-Video (T2V) Workflow

**Advantages**:
- Complete creative control
- No source material needed
- Best for abstract or complex scenes

**Challenges**:
- Requires detailed prompts
- Longer generation times
- Needs high/low noise model pairs

### Text+Image-to-Video (TI2V-5B) Hybrid

**Fastest Option**:
- 5B model runs on 8GB VRAM
- Supports both T2V and I2V
- 720P at 24fps

**Trade-offs**:
- Slightly lower quality than 14B models
- Less complex motion capability
- Excellent for rapid prototyping

## Troubleshooting

### Common Issues

**Out of Memory Errors**:
- Switch to lower quantization (Q4_K_S)
- Reduce resolution
- Enable model offloading
- Close other VRAM-heavy applications
- Increase system page file size

**Slow Generation**:
- Use Lightning LoRA
- Install Sage Attention
- Lower step count (20 → 15)
- Consider Q4 over Q6

**Poor Quality Output**:
- Increase quantization level (Q4 → Q5/Q6)
- Improve prompt detail
- Adjust CFG scale (test 5.0-8.0)
- Check for correct high/low noise pairing

**Black or Corrupted Output**:
- Verify model files aren't corrupted
- Sage Attention may conflict (disable for testing)
- Check CUDA/PyTorch versions
- Ensure VAE is correct version

**First Frame Issues**:
- Use first-frame fix subgraph
- Enable in workflow templates
- Addresses VAE decoding artifacts

## Performance Benchmarks

### RTX A4000 Test Results

**Configuration**: Q5_K_M, 720P, 24fps, without Lightning

| Task | Resolution | Frames | Steps | Time | VRAM Peak |
|------|------------|--------|-------|------|-----------|
| T2V | 1280×720 | 73 | 20 | 14 min | 14.2 GB |
| I2V | 1280×720 | 73 | 20 | 13 min | 14.5 GB |
| S2V | 1024×704 | 97 | 20 | 16 min | 15.1 GB |
| TI2V-5B | 1280×704 | 73 | 20 | 8 min | 11.8 GB |

**With Lightning LoRA** (4 steps, CFG 1.0):
- T2V: ~4-5 minutes (65% faster)
- I2V: ~4 minutes (70% faster)
- Quality difference: Minimal for most use cases

**With Sage Attention 2.2**:
- Additional 30-40% speed improvement
- No quality degradation
- Best with Q5/Q6 quantization

## Best Practices Summary

### For RTX A4000 (16GB VRAM)

1. **Quantization**: Start with Q5_K_M, test Q6_K if RAM allows
2. **Lightning LoRA**: Essential for iteration speed
3. **Sage Attention**: Install for maximum efficiency
4. **Resolution**: 720P comfortable, 480P for longer videos
5. **System RAM**: 64GB+ recommended for smooth operation

### Workflow Optimization

1. Test prompts at 480P with Lightning LoRA first
2. Refine prompt and settings
3. Generate final at 720P without Lightning for best quality
4. Use I2V for controlled outputs, T2V for creative freedom
5. S2V for character animation with audio

### Quality vs Speed

**Maximum Quality**:
- Q8_0 or Q6_K quantization
- 720P resolution
- 20-30 steps, no Lightning
- CFG 7.0-7.5

**Balanced**:
- Q5_K_M quantization
- 720P resolution
- 4 steps with Lightning LoRA
- CFG 1.0

**Maximum Speed**:
- Q4_K_M quantization
- 480P resolution
- 4 steps with Lightning LoRA
- Sage Attention enabled

## Future Developments

### Wan2.2 Roadmap

- **Animate Character Models**: Video-to-video pose transfer
- **Extended Context**: Longer video generation
- **ControlNet Integration**: Depth, pose, edge guidance
- **LoRA Training**: Custom style/character training

### Hardware Trends

- **RTX 50xx Series**: Sage Attention 3 support for 2x additional speedup
- **Blackwell FP4 Cores**: 5x faster attention operations
- **Unified Memory**: Better handling of large models

## Conclusion

Wan2.2 GGUF quantization makes professional video generation accessible on consumer hardware like the RTX A4000. With proper configuration, prompt engineering, and optimization techniques, you can generate cinematic-quality videos locally without expensive cloud services.

Key takeaways:
- GGUF quantization is production-ready for most use cases
- Lightning LoRA provides massive speed improvements
- Sage Attention unlocks additional performance
- Detailed prompts yield significantly better results
- The 5B TI2V model offers the best speed/quality balance

Start with the TI2V-5B model and Lightning LoRA for rapid experimentation, then move to the 14B models with Q5/Q6 quantization for final production work.

## Resources

### Official Links
- GitHub: https://github.com/Wan-Video/Wan2.2
- Technical Paper: https://arxiv.org/abs/2503.20314
- Hugging Face: https://huggingface.co/Wan-AI/

### GGUF Models
- QuantStack Hub: https://huggingface.co/QuantStack
- City96 GGUF: https://huggingface.co/city96

### ComfyUI Resources
- ComfyUI Docs: https://docs.comfy.org/tutorials/video/wan/wan2_2
- ComfyUI Manager: Install via ComfyUI interface
- Example Workflows: https://comfyanonymous.github.io/ComfyUI_examples/wan22/

### Community
- Discord: https://discord.gg/AKNgpMK4Yj
- Reddit: r/comfyui
- CivitAI: Community workflows and models