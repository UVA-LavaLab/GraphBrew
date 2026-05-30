# Gate 256 — ECG profile registry

Status: **active**

## Totals

- manifest_profiles: 30
- stages: 30
- citations_total: 25
- distinct_citations: 12
- violations: 0

## Rules

- **R1** — every stage.profiles[*] token resolves to manifest.profiles
- **R2** — every manifest.profiles key has a non-empty description
- **R3** — every manifest.profiles key is referenced by stage, pytest, helper, or README (no dead profiles)
- **R4** — every --profile / profiles= citation outside the manifest resolves
- **R5** — profile names match ^[a-z][a-z0-9_]*$
- **R6** — stage names match ^[0-9]+[a-z0-9]*_[a-z][a-z0-9_]*$
- **R7** — each stage's profiles list is non-empty and duplicate-free

## Manifest profiles

- `available_cache_sim_ecg_pfx`
- `available_replacement`
- `available_sniper_replacement`
- `final_cache_sim`
- `final_cache_sim_ecg_pfx`
- `final_cache_sim_l_curve`
- `final_droplet`
- `final_replacement`
- `final_sniper_droplet`
- `final_sniper_replacement`
- `gem5_ecg_pfx_tiny_smoke`
- `local_cache_sim_diversity_medium`
- `local_cache_sim_diversity_smoke`
- `local_cache_sim_pfx_diversity_smoke`
- `rehearsal`
- `sniper_droplet_smoke`
- `sniper_kernel_smoke`
- `sniper_sift_benchmark_smoke`
- `sniper_sift_benchmark_suite`
- `sniper_sift_cit_patents_long`
- `sniper_sift_cit_patents_smoke`
- `sniper_sift_ecg_pfx_smoke`
- `sniper_sift_file_droplet_smoke`
- `sniper_sift_file_ecg_pfx_smoke`
- `sniper_sift_file_replacement_smoke`
- `sniper_sift_file_smoke`
- `sniper_sift_file_thread_smoke`
- `sniper_sift_replacement_smoke`
- `sniper_smoke`
- `sniper_thread_scaling`

## Stages (run-order)

- `01_cache_sim_component_proof`
- `02_gem5_replacement_rehearsal`
- `03_droplet_actual_edge_rehearsal`
- `04_sniper_pr_kernel_smoke`
- `05_sniper_thread_scaling_smoke`
- `06_sniper_kernel_smoke_suite`
- `07_sniper_droplet_kernel_smoke`
- `08_sniper_sift_benchmark_smoke`
- `09_sniper_sift_benchmark_suite`
- `09b_sniper_sift_replacement_smoke`
- `09c_sniper_sift_file_smoke`
- `09d_sniper_sift_file_replacement_smoke`
- `09e_sniper_sift_file_droplet_smoke`
- `09f_sniper_sift_file_thread_smoke`
- `09g_sniper_sift_ecg_pfx_smoke`
- `09g1_sniper_sift_file_ecg_pfx_pr_bfs_smoke`
- `09g2_sniper_sift_file_ecg_pfx_sssp_smoke`
- `09g_sniper_sift_cit_patents_smoke`
- `09h_sniper_sift_cit_patents_long`
- `09i_gem5_ecg_pfx_tiny_smoke`
- `10_cache_sim_large_replacement`
- `10b_cache_sim_l_curve`
- `11_cache_sim_large_ecg_pfx`
- `11a_cache_sim_available_ecg_pfx`
- `12_cache_sim_email_core_diversity`
- `12a_cache_sim_cit_patents_diversity`
- `12b_cache_sim_email_core_ecg_pfx_diversity`
- `15_gem5_available_replacement`
- `20_gem5_large_replacement`
- `30_gem5_large_droplet`

## Violations

None.
