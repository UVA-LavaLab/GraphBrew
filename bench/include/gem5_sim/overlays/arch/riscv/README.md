# GraphBrew RISC-V ECG Extract Overlay

This directory contains the initial gem5 ISA-description scaffold for the paper-style RISC-V custom instruction:

```text
ecg.extract rd, rs1
```

The scaffold reserves RISC-V custom-0 (`opcode=0x0b`, `OPCODE5=0x02`) with `FUNCT3=0` and `FUNCT7=0`. It decodes the paper-style fixed 64-bit layout from `rs1`: low 32 bits are the real vertex ID, bits 32-39 are DBG metadata, bits 40-47 are P-OPT metadata, and bits 48-63 are the ECG_PFX target hint. The real vertex ID is written to `rd`. The decoded metadata is stored in GraphBrew hint storage, and the ECG_PFX target is forwarded to the same prefetch-target hint queue used by the x86 m5ops prototype. If the PFX field is zero, the implementation preserves the original scaffold behavior by forwarding the real vertex ID as the target.

This is intentionally a scaffold, not a final paper claim path. Future work must connect decoded DBG/P-OPT fields to timing-visible replacement consumers and validate a RISC-V build/run.
