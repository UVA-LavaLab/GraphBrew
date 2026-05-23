# COrder — Algorithm ID: 10

## Citation

```bibtex
@inproceedings{corder-bigdata17,
  title     = {Making Caches Work for Graph Analytics},
  author    = {Zhang, Yunming and Kiriansky, Vladimir and Mendis, Charith and Amarasinghe, Saman and Zaharia, Matei},
  booktitle = {2017 IEEE International Conference on Big Data (IEEE Big Data)},
  year      = {2017}
}
```

## Official Repository

Referenced in DBG repo ([faldupriyank/dbg](https://github.com/faldupriyank/dbg)) as [3] and in IISWC'18 repo as [1].

## Key Contributions

1. **Cache-line-aware ordering**: Optimizes vertex placement for same-cache-line co-access
2. **Frequency-based reordering**: Identifies frequently co-accessed vertices and places them adjacently
3. **Low overhead**: Lightweight preprocessing that amortizes over multiple iterations

## GraphBrew Integration

- **Algorithm ID**: 10 (CORDER)
- **C++ Implementation**: `bench/include/graphbrew/reorder/reorder_classic.h`
- **External Library**: `bench/include/external/corder/` (bundled)
- **CLI**: `-o 10`
