This custom-built, lightweight AI model learns to compress and reconstruct the visual datasets it’s trained on — almost like how memory works in dreams. It doesn’t copy pixels; instead, it reimagines what it’s seen, creating new visuals that feel related to the originals.

You can guide it in two ways:

Reconstruction Mode: Feed it a single image or a small dataset and train for many epochs — it will try to recall and rebuild what it saw, producing a distorted yet recognizable version.
Generalization Mode: Train it on large datasets with fewer epochs — it begins to dream, generating unique, abstract images that blend patterns and memories from across the data.

APVD is built around a Variational Autoencoder (VAE) architecture, meaning it learns a compressed “recipe” for each image and then paints new ones from that recipe. The result is a surreal, memory-like reconstruction — a glimpse into how machines might dream.


================================================================================
APVD (AI Pixel Value Determinator) — What This Project Does
================================================================================

This program is a desktop tool (Python + Tkinter + PyTorch) that learns a
compressed "recipe" for your images, then invents new pictures that feel related
to what it saw — without copying pixels exactly. Many people use a loose,
poetic name for that kind of output: "synthetic dreams" or "artificial dreams."
This note explains what is actually happening under the hood and how that
metaphor fits (and where it does not).


--------------------------------------------------------------------------------
The core idea: a Variational Autoencoder (VAE)
--------------------------------------------------------------------------------

A VAE is a small neural network with two main parts:

  ENCODER — Looks at an image and summarizes it as a short list of numbers
            (a "latent vector," here 256 dimensions). Those numbers are not
            meant to be human-readable; they are coordinates in an abstract
            space the model learns.

  DECODER — Takes a latent vector and paints an image back out from it.

Training pushes the model to do two things at once:

  1) RECONSTRUCTION — When you feed a training image in, the decoded output
     should look like the original (as much as the small bottleneck allows).

  2) REGULARIZATION (KL term) — Latent codes are encouraged to stay in a
     smooth, continuous region so you can pick new points "between" or "near"
     learned examples and still get sensible images.

Loss in this project combines reconstruction (binary cross-entropy on pixel
values) plus that KL term — standard VAE training.


--------------------------------------------------------------------------------
What you feed it
--------------------------------------------------------------------------------

You can train on:

  • Individual image files
  • Every image in a folder (recursive)
  • Video files — frames are sampled (with a configurable stride and optional
    max frame count) and treated like extra still images

All inputs are resized to one training resolution (default 256×256, adjustable).
The convolutional encoder downsamples through several stages; the decoder builds
up from a small grid and upsamples to your chosen output size.


--------------------------------------------------------------------------------
How "variations" are made
--------------------------------------------------------------------------------

After training, the model does not store your dataset inside the weights in a
literal way. It has learned statistics and structure. New images come from
choosing a point in latent space and running the decoder.

Typical flow in this app:

  • Pick an "anchor" — often a random training image encoded to its mean
    latent (mu), or a weighted blend of several training images' latents
    ("Blend trained images").

  • Add controlled noise — scaled by "Variation Intensity." Larger values wander
    farther from the anchor; the result can look more abstract or surprising.

  • Optional "cleanup iterations" — decode → encode → decode a few times. That
    tends to pull the image back toward something the VAE finds more "typical,"
    which can sharpen or stabilize the look.

If there is no training data in memory, random latent noise can still be
decoded (the model will hallucinate in the style it learned from past training).


--------------------------------------------------------------------------------
Modes that matter for video / "dream" sequences
--------------------------------------------------------------------------------

  GENERATE UNIQUE — One-shot (or several side-by-side) samples from the current
  settings.

  CHAOS MODE — Randomizes intensity, blend count, and cleanup iterations, then
  generates — good for unexpected frames.

  AUTO-CYCLE — Periodically generates new samples on a timer.

  DREAM CYCLE (MORPH) — Instead of jumping to unrelated latents every frame, the
  app moves smoothly from one latent point to another using spherical linear
  interpolation (SLERP). That keeps motion through latent space at more even
  "speed" than straight line blending, so morphs can feel fluid — ideal for
  long, drifting sequences.

  BLEND TRAINED IMAGES — Averages several encoded training images (with random
  weights) before adding noise, so new frames often mix themes from your set.


--------------------------------------------------------------------------------
Why "dreams" is a metaphor — and an honest limit
--------------------------------------------------------------------------------

Human dreams weave memory, emotion, and nonsense at a scale biology still does
not fully explain. This tool is tiny by comparison: one small VAE, a limited
dataset, and a fixed latent size. The outputs are mathematical interpolations and
samples from a learned manifold — not a model of sleep or the unconscious.

Calling them "synthetic" or "artificial dreams" is fair as *poetry*: the visuals
can drift, merge, and surprise in ways that feel oneiric, especially in Dream
Cycle. It is also fair to say they are a *small-scale*, stylized cousin of that
idea — more like a compressed echo of your visual material than a scientific
simulation of dreaming.


--------------------------------------------------------------------------------
Practical summary
--------------------------------------------------------------------------------

APVD trains a convolutional VAE on your images (and optionally video frames),
then lets you explore its latent space with noise, blending, timed generation,
and smooth SLERP-based morphing. The result is original imagery biased toward
what you showed it — useful for art, backgrounds, and experimental video series
under names like Synthetic Dreams or Artificial Dreams, with the understanding
that the name evokes mood, not a claim about human sleep.





**Minimum Hardware Requirements (Small Datasets / Reconstruction Mode)
Designed for users who want to train APVD on low image counts (50–500 images) or run high‑epoch reconstructions.**

CPU: Intel Core i3 (or equivalent)

RAM: 8–16 GB

GPU:

Integrated graphics or

Entry‑level GPU with 2–4 GB VRAM

Training Performance:

Small datasets (4–20 images) train in 10–30 seconds

Great for testing, demos, and reconstruction‑focused experiments

Limitations:

Not suitable for large datasets

Generalization mode will be slow or may not run at all

Batch sizes must stay very small (1–4)




**## Supported GPUs (Generalization Mode)

### Recommended Minimum
- GTX 1660 SUPER
- GTX 1660
- GTX 1650 SUPER
- GTX 1650 GDDR6
- GTX 1060 6GB

### Borderline (Works with limitations)
- GTX 970 4GB

### Not Recommended
- GTX 1050 Ti
- GTX 1050
- GTX 960
**
