# Rendering

The benchmark uses persistent 320×240 camera products and the released RTX
capture settings. Camera names and image units are specified in the
[policy interface](policy-interface.md). Preserve camera mounts, image orientation,
lighting, warmup and frame timing when comparing policies. Do not substitute
viewport screenshots for policy observations.

GPU rendering can vary between processes. Compare decoded images with a stated
tolerance and preserve exact scene, configuration and asset identities. Rendering
agreement alone does not establish matching physics or task success.
