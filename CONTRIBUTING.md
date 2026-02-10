# Contributing to GraphBrew

We appreciate contributions from the community! GraphBrew is a graph reordering and benchmarking framework, and there are many ways to help improve it.

## Ways to Contribute

+ **Bug fixes** - Performance or correctness issues
+ **New reordering algorithms** - Add your own vertex reordering methods
+ **New graph benchmarks** - Additional graph kernels (e.g., MST, SCC, ALS)
+ **Support for additional input formats** - New graph file formats
+ **ML/Perceptron improvements** - Better weight training or feature extraction
+ **Documentation** - Wiki pages, code comments, examples
+ **Python tooling** - Analysis scripts, visualization, automation

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/GraphBrew.git
   cd GraphBrew
   ```
3. **Build the project**:
   ```bash
   make all
   ```
4. **Run tests**:
   ```bash
   make test
   ```

## Development Guidelines

### Code Style

This repo follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html). Key points:
- Use 2-space indentation
- Keep lines under 100 characters
- Use descriptive variable names
- Add comments for complex logic

### Adding New Algorithms

See the [Wiki: Adding New Algorithms](https://github.com/UVA-LavaLab/GraphBrew/wiki/Adding-New-Algorithms) for detailed instructions.

Quick overview:
1. Add your algorithm to `bench/include/gapbs/benchmark.h`
2. Register it in the `ReorderingAlgo` enum
3. Add training support in `scripts/lib/utils.py` (ALGORITHMS dict)

### Adding Algorithm Variants

Algorithm variant lists are defined in `scripts/lib/utils.py` as the **single source of truth**:
- `GRAPHBREW_VARIANTS` - GraphBrewOrder clustering variants  
- `RABBITORDER_VARIANTS` - RabbitOrder variants (csr, boost)

Other constants also centralized in `utils.py`:
- `ALGORITHMS` / `SLOW_ALGORITHMS` - Algorithm definitions
- `BENCHMARKS` - Benchmark list
- `SIZE_SMALL` / `SIZE_MEDIUM` / `SIZE_LARGE` / `SIZE_XLARGE` - Size thresholds (MB)
- `TIMEOUT_REORDER` / `TIMEOUT_BENCHMARK` / `TIMEOUT_SIM` / `TIMEOUT_SIM_HEAVY` - Timeouts (seconds)

**Never duplicate these definitions** in other files - always import from `utils.py`.

### Adding New Benchmarks

See the [Wiki: Adding New Benchmarks](https://github.com/UVA-LavaLab/GraphBrew/wiki/Adding-New-Benchmarks) for detailed instructions.

### Pull Request Process

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes with clear commit messages
3. Ensure all tests pass: `make test`
4. Push to your fork: `git push origin feature/my-feature`
5. Open a Pull Request against `main`

Our CI service (Travis) will automatically run sanity checks on your PR.

### Commit Messages

Use clear, descriptive commit messages:
```
Add: New RabbitOrder integration
Fix: Memory leak in Leiden community detection  
Update: Perceptron weights for PageRank workload
Docs: Add wiki page for correlation analysis
```

## Reporting Issues

When reporting bugs, please include:
- GraphBrew version / commit hash
- Operating system and compiler version
- Steps to reproduce the issue
- Expected vs actual behavior
- Relevant error messages or logs

## Feature Requests

Before starting a large feature:
1. Check existing [issues](https://github.com/UVA-LavaLab/GraphBrew/issues) for similar requests
2. Open an issue to discuss your proposal
3. Wait for feedback before investing significant time

## Questions?

- Check the [Wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki) for documentation
- Open an [issue](https://github.com/UVA-LavaLab/GraphBrew/issues) for questions
- See [FAQ](https://github.com/UVA-LavaLab/GraphBrew/wiki/FAQ) for common questions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).

---

Thank you for contributing to GraphBrew! 🍺
