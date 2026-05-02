# KV Cache Quantization Baseline (Intel Arc Pro B70)

## Grounding & Paper References

- **RotorQuant**: [RotorQuant: Clifford Algebra Vector Quantization](file:///d:/User%20Files/Desktop/RotorQuant/paper/rotorquant.md)
- **IsoQuant1**: [IsoQuant: Quaternion-based Isoclinic Rotation (Paper 1)](file:///d:/User%20Files/Desktop/RotorQuant/paper/isoquant%20paper%201.md)
- **IsoQuant2**: [IsoQuant: Hardware-Aligned SO(4) Isoclinic Rotations (Paper 2)](file:///d:/User%20Files/Desktop/RotorQuant/paper/isoquant%20paper%202.md)

## Performance & VRAM scaling by Context Length

| K Type | V Type | Context | Prefill (TK/s) | Decode (TK/s) | KV Size (MiB) | Context VRAM (MiB) | Total VRAM (MiB) | PPL | Coherency | Needle |
|--------|--------|---------|----------------|---------------|---------------|--------------------|------------------|-----|-----------|--------|
| F16 | F16 | 512 | 0.00 | 0.00 | 0.00 | 4108.00 | 31906.00 | 0.00 | FAILED | FAILED |
| F16 | Q8_0 | 512 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| F16 | ISO4 | 512 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| F16 | ROTOR4 | 512 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| Q8_0 | F16 | 512 | 0.00 | 0.00 | 0.00 | 4103.00 | 31906.00 | 0.00 | FAILED | FAILED |
| Q8_0 | Q8_0 | 512 | 0.00 | 0.00 | 0.00 | 4101.00 | 31906.00 | 0.00 | FAILED | FAILED |
| Q8_0 | ISO4 | 512 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| Q8_0 | ROTOR4 | 512 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ISO4 | F16 | 512 | 0.00 | 0.00 | 0.00 | 4102.00 | 31906.00 | 0.00 | PARTIAL (1/3) | FAILED |
| ISO4 | Q8_0 | 512 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ISO4 | ISO4 | 512 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ISO4 | ROTOR4 | 512 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ROTOR4 | F16 | 512 | 0.00 | 0.00 | 0.00 | 4102.00 | 31906.00 | 0.00 | FAILED | FAILED |
| ROTOR4 | Q8_0 | 512 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ROTOR4 | ISO4 | 512 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ROTOR4 | ROTOR4 | 512 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| F16 | F16 | 4096 | 0.00 | 0.00 | 0.00 | 4150.00 | 31906.00 | 0.00 | FAILED | FAILED |
| F16 | Q8_0 | 4096 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| F16 | ISO4 | 4096 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| F16 | ROTOR4 | 4096 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| Q8_0 | F16 | 4096 | 0.00 | 0.00 | 0.00 | 4135.00 | 31906.00 | 0.00 | FAILED | FAILED |
| Q8_0 | Q8_0 | 4096 | 0.00 | 0.00 | 0.00 | 4124.00 | 31906.00 | 0.00 | FAILED | FAILED |
| Q8_0 | ISO4 | 4096 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| Q8_0 | ROTOR4 | 4096 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ISO4 | F16 | 4096 | 0.00 | 0.00 | 0.00 | 4129.00 | 31906.00 | 0.00 | PARTIAL (1/3) | FAILED |
| ISO4 | Q8_0 | 4096 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ISO4 | ISO4 | 4096 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ISO4 | ROTOR4 | 4096 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ROTOR4 | F16 | 4096 | 0.00 | 0.00 | 0.00 | 4129.00 | 31906.00 | 0.00 | FAILED | FAILED |
| ROTOR4 | Q8_0 | 4096 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ROTOR4 | ISO4 | 4096 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ROTOR4 | ROTOR4 | 4096 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| F16 | F16 | 16384 | 0.00 | 0.00 | 0.00 | 4294.00 | 31906.00 | 0.00 | FAILED | FAILED |
| F16 | Q8_0 | 16384 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| F16 | ISO4 | 16384 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| F16 | ROTOR4 | 16384 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| Q8_0 | F16 | 16384 | 0.00 | 0.00 | 0.00 | 4245.00 | 31906.00 | 0.00 | FAILED | FAILED |
| Q8_0 | Q8_0 | 16384 | 0.00 | 0.00 | 0.00 | 4200.00 | 31906.00 | 0.00 | FAILED | FAILED |
| Q8_0 | ISO4 | 16384 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| Q8_0 | ROTOR4 | 16384 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ISO4 | F16 | 16384 | 0.00 | 0.00 | 0.00 | 4221.00 | 31906.00 | 0.00 | PARTIAL (1/3) | FAILED |
| ISO4 | Q8_0 | 16384 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ISO4 | ISO4 | 16384 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ISO4 | ROTOR4 | 16384 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ROTOR4 | F16 | 16384 | 0.00 | 0.00 | 0.00 | 4221.00 | 31906.00 | 0.00 | FAILED | FAILED |
| ROTOR4 | Q8_0 | 16384 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ROTOR4 | ISO4 | 16384 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ROTOR4 | ROTOR4 | 16384 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| F16 | F16 | 32768 | 0.00 | 0.00 | 0.00 | 4482.00 | 31906.00 | 0.00 | FAILED | FAILED |
| F16 | Q8_0 | 32768 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| F16 | ISO4 | 32768 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| F16 | ROTOR4 | 32768 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| Q8_0 | F16 | 32768 | 0.00 | 0.00 | 0.00 | 4535.00 | 31906.00 | 0.00 | FAILED | FAILED |
| Q8_0 | Q8_0 | 32768 | 0.00 | 0.00 | 0.00 | 4302.00 | 31906.00 | 0.00 | FAILED | FAILED |
| Q8_0 | ISO4 | 32768 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| Q8_0 | ROTOR4 | 32768 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ISO4 | F16 | 32768 | 0.00 | 0.00 | 0.00 | 4487.00 | 31906.00 | 0.00 | PARTIAL (1/3) | FAILED |
| ISO4 | Q8_0 | 32768 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ISO4 | ISO4 | 32768 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ISO4 | ROTOR4 | 32768 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ROTOR4 | F16 | 32768 | 0.00 | 0.00 | 0.00 | 4487.00 | 31906.00 | 0.00 | FAILED | FAILED |
| ROTOR4 | Q8_0 | 32768 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ROTOR4 | ISO4 | 32768 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
| ROTOR4 | ROTOR4 | 32768 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | FAILED | FAILED |
