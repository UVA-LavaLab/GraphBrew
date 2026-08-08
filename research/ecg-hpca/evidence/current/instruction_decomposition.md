# Instruction-Count Interpretation

**Status:** open attribution question; does not invalidate the measured
complete-design STOP.

The completed no-prefetch screen used equal semantic work and one build, but
K2 retired fewer instructions than conventional policies:

| Graph | K2/LRU retired-instruction ratio |
|---|---:|
| web-Google-n16 | 0.8546 |
| soc-pokec-n16 | 0.8144 |
| cit-Patents-n18-sym | 0.9120 |
| **geomean** | **0.8594** |

Instruction inequality is allowed for a declared **complete design** that
includes record layout, transport, ISA, StreamShield, and replacement. It is not
evidence that K2's replacement rule alone beats another policy.

The clean replacement attribution inside the K2 family is:

- K2-RRIP+StreamShield versus K2-LRU+StreamShield;
- same record stream, delivery path, ISA, and instruction count;
- aggregate time ratio 0.9330 and off-chip ratio 0.7855.

Before promoting a complete-design speedup claim, classify the K2/conventional
instruction delta on one bounded PageRank cell. The required decomposition is:

1. conventional edge load plus ordinary property load;
2. compact replacement record plus ordinary property load with matched loop
   shape;
3. compact record plus K2-M property load;
4. K2-LRU+StreamShield versus K2-RRIP+StreamShield.

The investigation must distinguish intended record/ISA work reduction from
asymmetric compiler or loop specialization. Measured target time must not be
counterfactually divided by instruction count.
