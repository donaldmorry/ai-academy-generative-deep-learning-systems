# Chapter 11: Music Generation

> **Source:** *Generative Deep Learning (2nd ed.) — David Foster*, Chapter 11
> **Part:** Part III — Applications
> **Estimated time:** 10–12 hours
> **Prerequisites:** Course 4, Chapters 04 and 09 — GANs and transformer training; Course 1, Chapter 13 — sequence data preprocessing concepts

---

## Chapter Overview

Music generation introduces unique challenges: polyphonic structure, temporal coherence, and multiple simultaneous voices. You will parse MIDI files from the Bach Cello Suite dataset, tokenize musical events, and train a transformer for monophonic generation with sine positional encoding. Polyphonic tokenization extends to MuseGAN—a GAN framework with generator and critic for Bach chorales. This chapter applies sequence and adversarial techniques from Parts II–III to a creative, multi-track domain.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Parse MIDI files and represent musical events as token sequences for model input
2. Design tokenization schemes for monophonic and polyphonic music
3. Build a music-generating transformer with sine positional encoding in Keras
4. Handle multiple input/output streams for polyphonic music representation
5. Implement MuseGAN's generator and critic for multi-track Bach chorale generation
6. Evaluate generated music for coherence, harmonic structure, and diversity
7. Compare transformer vs GAN approaches for symbolic music generation

---

## Sections

| # | Section | Topics |
|---|--------|--------|
| 11.1 | [Music Generation Introduction](./section-01-music-generation-introduction.md) | Symbolic vs audio; MIDI format; challenges |
| 11.2 | [Parsing MIDI Files](./section-02-parsing-midi-files.md) | Bach Cello Suite; note events; tempo; track structure |
| 11.3 | [Tokenization](./section-03-tokenization.md) | Event-based tokens; vocabulary; sequence construction |
| 11.4 | [Music Transformer](./section-04-music-transformer.md) | Training set creation; model architecture; sine position encoding |
| 11.5 | [Training & Analysis](./section-05-training-and-analysis.md) | Monophonic generation; listening evaluation; metrics |
| 11.6 | [Polyphonic Tokenization](./section-06-polyphonic-tokenization.md) | Multi-track representation; piano roll encoding |
| 11.7 | [MuseGAN Architecture](./section-07-musegan-architecture.md) | Generator; critic; bar-level generation; Bach chorales |
| 11.8 | [MuseGAN Training & Analysis](./section-08-musegan-training-and-analysis.md) | Adversarial training; multi-track output evaluation |

---

## Lab / Project

See also: [Lab 11](./section-lab-11-music-transformer-and-musegan.md)

**Lab 11: Music Transformer & MuseGAN**

1. Parse Bach Cello Suite MIDI; tokenize and train a music transformer in Keras.
2. Generate 3 monophonic melodies; export as MIDI and document listening notes.
3. Prepare Bach chorale piano-roll data for MuseGAN.
4. Train or run a pretrained MuseGAN; generate a 4-bar polyphonic chorale.
5. *Deliverable:* MIDI files for transformer and MuseGAN outputs, with brief quality assessment.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Sequence transformers | GPT architecture (Course 4, Chapter 09) |
| GAN generator/critic | Adversarial training (Course 4, Chapter 04) |
| Structured data generation | Autoregressive factorization (Course 4, Chapter 05) |

---

## Prerequisites

- Course 4, Chapters 04 and 09 — GANs and transformer training
- Course 1, Chapter 13 — sequence data preprocessing concepts

---

## Self-Assessment

1. What additional challenges does music generation pose compared to text generation?
2. How are MIDI events converted to token sequences?
3. What is sine positional encoding, and why is it used for music?
4. How does MuseGAN represent polyphonic music differently from the monophonic transformer?
5. What are the generator and critic roles in MuseGAN?
6. How would you evaluate the quality of generated music?

---

**Previous:** [Chapter 10 — Advanced GANs](../chapter-10-advanced-gans/README.md)
**Next:** [Chapter 12 — World Models](../chapter-12-world-models/README.md)
