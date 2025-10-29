# Chroma Radiance: Experimental Model Without VAE

- by Daniel Sandner, 2025, sketch for sandner.art article

## Introduction: The Dawn of Pixel-Space Generation

The AI image generation landscape is experiencing a paradigm shift with the introduction of **Chroma1-Radiance**, an experimental model that challenges the fundamental architecture of traditional diffusion models. Developed by the LodestoneRock team and released under the permissive Apache 2.0 license, Chroma Radiance represents a radical departure from conventional approaches by operating entirely in pixel space, eliminating the need for a Variational Autoencoder (VAE).

This groundbreaking approach addresses one of the most persistent issues in AI image generation: VAE compression artifacts. By generating images directly in pixel space rather than latent space, Chroma Radiance promises clearer outputs, more accurate live previews, and a simplified architecture that could reshape how we think about image synthesis.

## Understanding the Chroma Model Family

Before diving into Radiance specifically, it's essential to understand the broader Chroma ecosystem. The Chroma model family represents a massive computational achievement, requiring over **105,000 hours of H100 GPU training time** and resulting in a robust 8.9 billion parameter system.

### The Chroma Series Overview

**Chroma1-Base (512x512)**
- Core foundational model trained at 512x512 resolution
- 8.9B parameters based on FLUX.1-schnell architecture
- Designed as a neutral starting point for fine-tuning
- Trained on 5 million carefully curated images from a 20 million sample pool
- No aesthetic tuning or DPO (Direct Preference Optimization) applied

**Chroma1-HD (1024x1024)**
- High-resolution fine-tune of Chroma1-Base
- Optimized for 1024x1024 generation
- Ideal starting point for high-resolution LoRA training
- Maintains the same parameter count with resolution-specific optimization

**Chroma1-Flash**
- Experimental speed-optimized variant
- Uses "baked" CFG (Classifier-Free Guidance) for faster inference
- Explores acceleration techniques for flow-matching models
- No GAN-based training involved
- Delta weights can be applied to other Chroma versions

**Chroma1-Radiance (Work in Progress)**
- Revolutionary pixel-space model
- Eliminates VAE compression artifacts entirely
- True end-to-end generation without latent space encoding
- Currently in active development requiring extended training
- Available in FP8, GGUF, and standard formats

## Architectural Innovations

### Key Technical Modifications

The Chroma family incorporates several significant architectural improvements over the base FLUX.1 model:

**1. Efficient Timestep Encoding**
The original FLUX.1-schnell model contained a 3.3 billion parameter timestep-encoding layer that encoded only a single value. The Chroma team replaced this oversized component with a streamlined 250 million parameter Feed-Forward Network (FFN), saving substantial computational resources while maintaining accuracy.

**2. MMDiT Masking**
Chroma implements sophisticated attention masking to address issues with padding tokens in the T5 text encoder. By masking T5 padding tokens, the model prevents attention drift and ensures focus on relevant prompt elements, enhancing fidelity and training stability.

**3. Custom Timestep Sampling**
The team implemented a custom timestep sampling distribution (-x²) to prevent loss spikes during training. This ensures the model trains effectively across both high-noise and low-noise regions, resulting in more consistent quality.

**4. Rectified Flow Transformer**
Built on the powerful rectified flow transformer architecture, Chroma processes text through a T5 XXL text encoder and generates images using flow-matching techniques rather than traditional diffusion.

### The Radiance Revolution: Pixel Space Generation

**What Makes Radiance Different?**

Traditional diffusion models follow a two-stage process:
1. Generate images in a compressed latent space
2. Decode from latent space to pixel space using a VAE

This process introduces inevitable information loss. The VAE must estimate what the final image should look like, leading to:
- Compression artifacts
- Detail loss
- Color inaccuracies
- Blurring in fine details

Chroma Radiance eliminates this bottleneck by generating directly in pixel space. This means:
- **No VAE required**: The model generates RGB pixels directly
- **No compression loss**: What the model generates is what you see
- **Accurate live previews**: Since generation happens in pixel space, previews show the actual output
- **Reduced complexity**: Fewer components in the pipeline means fewer potential failure points

**Technical Implementation**

Based on the PixelNerd (Pixel Neural Field Diffusion) approach, Radiance models patch-wise decoding with neural fields, creating a single-scale, single-stage, efficient, end-to-end solution. This represents the first practical and high-quality pixel space diffusion model.

The workflow in ComfyUI uses `EmptyChromaRadianceLatentImage` instead of traditional latent image nodes, and no VAE loader is required. The model directly outputs to pixel space throughout the generation process.

## Model Comparison Table

| Feature | Chroma Radiance | FLUX.1-dev | Qwen-Image | SD3.5 Large |
|---------|----------------|------------|------------|-------------|
| **Parameters** | 8.9B | 12B | 20B | 8B |
| **Architecture** | Pixel-Space DiT | Rectified Flow Transformer | MMDiT | MMDiT |
| **VAE Required** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **License** | Apache 2.0 | Non-commercial (dev) | Apache 2.0 | Community License |
| **Native Resolution** | 512-1024px | 1024x1024 | 1328x1328 | Up to 1 megapixel |
| **Training Data** | 5M curated | Proprietary | Extensive multimodal | Proprietary |
| **Text Encoder** | T5 XXL | T5 XXL + CLIP | Qwen2.5-VL + T5 | T5 XXL + CLIP |
| **Compression Artifacts** | None (pixel space) | Minimal | Present | Present |
| **Fine-tuning Ease** | High (base model) | Moderate | Moderate | Moderate |
| **Inference Speed** | Moderate | Fast (schnell variant) | Moderate | Moderate-Fast (turbo) |
| **Commercial Use** | ✅ Yes | ❌ No (dev) | ✅ Yes | ⚠️ Restricted |
| **LoRA Support** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **GGUF Available** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Status** | WIP | Stable | Stable | Stable |

**Notes:**
- **Chroma Radiance**: Best for VAE-free generation and fine-tuning experimentation. Still in development but shows promising results for eliminating compression artifacts.
- **FLUX.1**: Excellent prompt adherence and efficiency. The schnell variant offers the fastest inference but dev version is non-commercial.
- **Qwen-Image**: Superior benchmark performance (GenEval, DPG) and larger native resolution. Best for complex scene generation and text rendering.
- **SD3.5**: Strong heritage from Stable Diffusion ecosystem. Medium variant extends to 2 megapixels. Good for traditional workflows.

## Running Chroma Radiance GGUF on RTX A4000 (16GB)

### Hardware Profile: RTX A4000

The NVIDIA RTX A4000 with 16GB VRAM represents a professional-grade GPU well-suited for AI workloads:
- **VRAM**: 16GB GDDR6
- **CUDA Cores**: 6,144
- **Tensor Cores**: 48 (3rd Gen)
- **Memory Bandwidth**: 448 GB/s
- **Power Draw**: 140W TDP

This configuration provides a sweet spot for running quantized models with good performance.

### GGUF Quantization Levels Explained

GGUF (GPT-Generated Unified Format) quantization reduces model precision to fit larger models into limited VRAM:

| Quantization | File Size (approx) | VRAM Usage | Quality | Recommendation |
|--------------|-------------------|------------|---------|----------------|
| **Q2_K** | ~3.0 GB | ~4.5 GB | Low | Avoid unless extremely limited |
| **Q3_K_M** | ~4.0 GB | ~6.0 GB | Fair | Emergency use only |
| **Q4_K_M** | ~5.2 GB | ~7.8 GB | Good | ⭐ Recommended for 8GB GPUs |
| **Q5_K_M** | ~6.3 GB | ~9.5 GB | Very Good | ⭐ Recommended for 12GB GPUs |
| **Q6_K** | ~7.3 GB | ~11.0 GB | Excellent | ⭐ Best for 16GB GPUs |
| **Q8_0** | ~9.5 GB | ~14.2 GB | Near-lossless | Optimal for RTX A4000 |
| **FP16/F16** | ~18.0 GB | ~20+ GB | Perfect | Requires 24GB+ VRAM |

**For RTX A4000 (16GB)**: Q8_0 or Q6_K are optimal choices, offering near-original quality while fitting comfortably in VRAM.

### Setup Instructions

**1. Install ComfyUI-GGUF Custom Node**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF
cd ComfyUI-GGUF
pip install --upgrade gguf
```

**2. Download Model Files**

Place files in the following directories:
- **GGUF Model**: `ComfyUI/models/unet/` or `diffusion_models/`
- **CLIP Encoders**: `ComfyUI/models/clip/`

Recommended downloads for 16GB VRAM:
- `chroma1-radiance-Q8_0.gguf` (best quality) or `chroma1-radiance-Q6_K.gguf` (more headroom)
- No VAE needed!
- No separate CLIP needed (uses standard FLUX CLIP)

**3. ComfyUI Workflow Setup**

Key nodes for Chroma Radiance:
- **Unet Loader (GGUF)**: Load your quantized Radiance model
- **EmptyChromaRadianceLatentImage**: Create pixel-space "latent" (not actually latent)
- **KSampler**: Standard sampling with recommended settings
- **SaveImage**: Output directly to image (no VAE decode needed!)

### Optimization Tips for Maximum Performance

**Memory Management**

1. **Close Background Applications**: Free up system RAM and VRAM
   ```bash
   # Monitor VRAM usage
   nvidia-smi -l 1
   ```

2. **Enable Model Offloading**: If running low on VRAM
   - Use `enable_model_cpu_offload()` in Diffusers
   - ComfyUI automatically manages memory well

3. **Batch Size**: Start with 1, increase if stable
   - Single images: Safest approach
   - Batch of 2-4: Possible with Q6_K on 16GB

**Sampling Parameters**

Recommended starting values for Chroma Radiance:
- **Steps**: 30-40 (Chroma requires more steps than FLUX schnell)
- **CFG Scale**: 2.5-3.5 (higher values for stronger prompt adherence)
- **Sampler**: Euler, DPM++ 2M, or DDIM
- **Scheduler**: Normal or Karras

**Speed Optimizations**

1. **Use xFormers or Flash Attention**:
   ```bash
   pip install xformers
   # ComfyUI will auto-detect and use
   ```

2. **Enable Torch Compile** (ComfyUI 0.3.60+):
   - Provides 10-20% speed boost after initial compile
   - Set in settings or launch with `--highvram` flag

3. **Quantization Selection**:
   - Q6_K: Best balance for A4000 (11GB loaded, generates fast)
   - Q8_0: Maximum quality (14GB loaded, slightly slower)

**Stability Tips**

1. **Clear VRAM Between Generations**:
   - ComfyUI: Click "Free All Memory" button
   - Or restart workflow

2. **Monitor Temperature**:
   - RTX A4000 runs cool, but sustained loads can affect performance
   - Ensure adequate case airflow

3. **Update Drivers**: Use latest NVIDIA Studio or Game Ready drivers

### Expected Performance

On RTX A4000 with GGUF Q8_0:
- **Generation Time**: 2.5-4 minutes per 1024x1024 image (30-40 steps)
- **VRAM Usage**: 14-15GB peak
- **Simultaneous Generations**: 1-2 images comfortably
- **Quality**: Near-identical to FP16, no VAE artifacts

With Q6_K:
- **Generation Time**: 2-3 minutes per image
- **VRAM Usage**: 11-12GB peak
- **Simultaneous Generations**: 2-3 images possible
- **Quality**: Excellent, minimal perceptible difference from Q8

## Prompting Techniques for Chroma Models

### Understanding Chroma's Prompt Structure

Chroma models were trained on a diverse dataset including artistic, photographic, and niche styles. The prompting approach differs slightly from other models:

**1. Detailed, Natural Language Works Best**

Chroma responds well to descriptive, natural language prompts rather than comma-separated tags:

✅ **Good**: "A hyperrealistic macro photograph of a team of tiny bakers—each precisely 2 inches tall—collaborating on an enormous, golden-brown croissant with flaky, layered textures."

❌ **Less Effective**: "tiny bakers, croissant, macro, hyperrealistic, 8k, detailed"

**2. Specify Technical Details**

Include photography/artistic terms:
- Lighting: "Natural light", "Soft ambient light", "Cinematic lighting"
- Camera: "Close-up portrait", "Macro photograph", "Wide-angle shot"
- Style: "Amateur photography", "Professional studio shot", "Documentary style"

**3. Structured Prompt Template**

```
[Subject description] + [Action/Pose] + [Setting/Environment] + 
[Technical details] + [Lighting] + [Style/Aesthetic]
```

Example:
```
A solemn young woman with striking amber-orange eyes and pale complexion 
dusted with freckles, wearing an intricately embroidered high-collared 
black garment with gold brocade. Braided dark hair interwoven with gold 
and copper fabric. Soft warm lighting highlighting skin texture and detail 
against a muted grey-blue background. Hyperrealistic portrait photography.
```

**4. Negative Prompts**

Chroma benefits from negative prompts to avoid common issues:
```
low quality, ugly, unfinished, out of focus, deformed, disfigured, 
blurry, smudged, restricted palette, flat colors
```

### Caption vs Tag Style

**Caption Style (Recommended for Chroma)**:
Uses complete sentences and natural descriptions. This aligns better with Chroma's T5 training:

```
This is a nature documentary close-up photograph of the right side of 
the face of a tiger. The photograph is centered on its highly detailed 
and speckled eye surrounded by intricately detailed fur. Overlaid at the 
center is title text that says "RADIANCE" in large white 3D letters. 
Amateur photography. Unfiltered. Real life. Natural light. Subtle shadows.
```

**Tag Style (Less Optimal)**:
```
tiger eye, close-up, macro, detailed fur, speckled, text overlay, 
RADIANCE, 3D text, documentary, natural light, amateur, unfiltered
```

The caption style provides better context and relationships between elements, leading to more coherent compositions.

### Advanced Techniques

**1. Emphasis with Repetition**
Instead of weight syntax, repeat important terms:
```
A highly detailed, ultra-detailed, meticulously detailed sculpture...
```

**2. Context Layering**
Build context progressively:
```
[Main subject] + [Immediate environment] + [Background elements] + 
[Atmosphere/mood] + [Technical execution]
```

**3. Avoiding Over-specification**
Chroma can become confused with contradictory or excessive details. Keep prompts focused:
- ✅ 2-4 key descriptive sentences
- ❌ Paragraph-long exhaustive descriptions

**4. Text Generation**
Chroma handles text in images relatively well. For best results:
```
...with large white text that reads "YOUR TEXT" in bold 3D letters...
```

Be specific about text appearance: size, color, style, positioning.

## Troubleshooting Common Issues

### Black/Corrupted Images

**Causes & Solutions:**
1. **Insufficient VRAM**: Drop to lower quantization (Q6_K → Q5_K_M)
2. **Too High CFG**: Reduce CFG scale to 2.5-3.0
3. **Incompatible Sampler**: Switch to Euler or DPM++ 2M

### Slow Generation Times

**Optimization Steps:**
1. Enable xFormers/Flash Attention
2. Use Q6_K instead of Q8_0
3. Reduce steps to 25-30 for testing
4. Close other VRAM-intensive applications

### Model Loading Errors

**Check:**
1. Correct file placement in `models/unet/`
2. Latest ComfyUI version (0.3.60+)
3. ComfyUI-GGUF custom node installed
4. Sufficient disk space for temporary files

### Quality Issues

**If outputs lack detail:**
1. Increase steps to 35-40
2. Try Q8_0 instead of lower quantization
3. Enhance prompt specificity
4. Adjust CFG scale (test 2.5, 3.0, 3.5)

## Current Status and Future Development

### Radiance Development Timeline

**Current Stage (v0.1-0.2)**:
- Functional pixel-space generation
- Proof of concept achieved
- Quality approaching latent-space models
- Training ongoing for convergence

**Known Limitations**:
- Requires 10x more training for full detail convergence
- Limited fine-tuning examples available
- Fewer community resources than FLUX/SD
- Still experimental—expect updates

**Future Outlook**:
- Extended training to improve detail rendering
- LoRA compatibility improvements
- ControlNet support development
- Community fine-tunes emergence

### Community Reception

Chroma Radiance has been described as "one of the most exciting things happening in the space right now" (Ostris) due to its revolutionary architecture. The model demonstrates that pixel-space generation is not only viable but potentially superior for certain use cases.

## Conclusion

Chroma1-Radiance represents a bold experiment in diffusion model architecture, challenging the decade-long assumption that latent space compression is necessary for practical image generation. By eliminating the VAE bottleneck, Radiance offers:

- **Artifact-free generation**: No compression-related quality loss
- **Simplified architecture**: Fewer moving parts, more predictable behavior
- **True WYSIWYG**: Live previews match final output exactly
- **Fine-tuning potential**: Pure pixel-space training for specialized models

For users with 16GB VRAM (like RTX A4000 owners), Chroma Radiance GGUF offers an accessible entry point into this cutting-edge technology. The Q6_K and Q8_0 quantizations provide excellent quality while maintaining reasonable generation speeds.

As the model continues development and receives extended training, we can expect Radiance to mature into a formidable alternative to traditional latent-space models, potentially setting a new standard for diffusion-based image generation.

## Resources

### Official Links
- **Chroma1-Radiance**: https://huggingface.co/lodestones/Chroma1-Radiance
- **Chroma1-HD**: https://huggingface.co/lodestones/Chroma1-HD
- **ComfyUI Blog Post**: https://blog.comfy.org/p/humo-and-chroma1-radiance-support

### Tools & Extensions
- **ComfyUI-GGUF**: https://github.com/city96/ComfyUI-GGUF
- **GGUF Quantizations**: https://huggingface.co/QuantStack/Chroma1-HD-GGUF

### Community & Discussion
- **CivitAI Chroma Hub**: https://civitai.com/models/1330309/chroma
- **Technical Report**: Coming soon from LodestoneRock team

---

*Article Version: 1.0 | Last Updated: October 2025 | Testing planned on RTX A4000 16GB with ComfyUI*