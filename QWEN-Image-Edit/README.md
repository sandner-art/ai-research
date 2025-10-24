# Qwen Image and Edit: Local GGUF Generations and Editing with Lightning

Author: Daniel Sandner, for article on https://sandner.art/ (workflows and research)

## Research Document for Article Development

---

## Model Overview & Capabilities

### Qwen-Image
**Released:** August 4, 2025  
**Architecture:** 20B parameter Multimodal Diffusion Transformer (MMDiT)  
**License:** Apache 2.0 (fully open-source)

#### Key Capabilities:
- **Superior Text Rendering**: Excels at complex text rendering including multi-line layouts, paragraph-level semantics, and fine-grained details
- **Multilingual Support**: Exceptional performance in both alphabetic (English) and logographic (Chinese, Japanese, Korean, Italian) languages
- **Diverse Artistic Styles**: From photorealistic to impressionist paintings, anime aesthetics to minimalist design
- **Seven Aspect Ratios**: Supports 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3
- **32,000 Token Prompt Window**: Inherited from Qwen 2.5-VL backbone (vs. ~75 tokens in Stable Diffusion CLIP)
- **Resolution Support**: 256p to 1328p with dynamic resolution processing

### Qwen-Image-Edit
**Released:** August 18, 2025  
**Updated:** September 22, 2025 (Qwen-Image-Edit-2509)

#### Key Capabilities:
- **Precise Text Editing**: Bilingual (Chinese/English) text editing with preservation of original font, size, and style
- **Dual Semantic and Appearance Editing**: 
  - Low-level appearance editing (add/remove/modify elements while keeping other regions unchanged)
  - High-level semantic editing (IP creation, object rotation, style transfer with semantic consistency)
- **Multi-Image Editing** (2509 version): Supports 1-3 input images for person+person, person+product, person+scene combinations
- **Enhanced Consistency**: Improved facial identity preservation, product consistency, and text formatting
- **Novel View Synthesis**: 90° and 180° object rotation capabilities
- **ControlNet Support**: Depth maps, edge maps, keypoint maps, and more