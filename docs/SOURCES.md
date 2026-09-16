# Sources and Licenses

Snapshot date: 2026-09-16. No original Git history is included.

| Source key / destination | Original project | Version context | License context |
|---|---|---|---|
| `symm` / `.` | ispy2-symmflow3d | 2026-09-16 local source snapshot | Apache-2.0; adapted VQ code CC-BY-NC-4.0 |
| `ispy2` / `workflows/first_post_three_phase` | MeWM-ISPY2-DCE0-BiFlowNet-RF | feature/ispy2-dce0-biflow-rf | CC-BY-NC-4.0 |
| `mewm` / `workflows/roi32` | MeWM-ISPY2 | feature/ispy2-last-phase-diffusion | CC-BY-NC-4.0 |
| `pillar` / `support/pillar` | longitudinal_temporal_pillar | main; data helpers and downstream pCR evaluation | Existing source attribution retained; no new blanket license |
| `ispy2_codec` / `support/ispy2_codec` | MeWM-ISPY2-DCE0-Post-VQGAN | configuration dependency | MeWM-derived configuration |
| `mu_codec` / `support/mu_codec` | MeWM-MU-Glioma-Post-VQGAN | configuration dependency | MeWM-derived configuration |
| `registration` / `support/registration` | Quantitative-MRI-of-breast-cancer-patients-to-forecast-response-to-therapy-registration-t0-python | Python transform-reader source snapshot | MIT |

## Packaging Changes

- Original Python module names, model mathematics, losses and tensor contracts are retained.
- Absolute source and artifact paths are resolved through `research_release.py` and private local settings.
- Old SSH-script introspection is replaced by explicit local SSH settings.
- Historical identity values and cohort exclusions are not emitted into release configs; real asset checks remain strict.
- Configuration unit tests use temporary synthetic assets and retain mismatch/rejection assertions.
- First-post/three-phase model construction reads architecture settings without requesting unrelated codec identities.
- ROI32 reporting converts BF16 arrays to FP32 before NumPy; its numerical regression is covered by a CPU test.
- No source from the original running worktrees is modified. No patient artifacts or restricted weights are included.

## Attribution

MeWM-derived code: https://github.com/scott-yjyang/MeWM . Original license texts remain beside each included source tree.
SymmFlow original copyright and the adapted VQ exception remain in its LICENSE and third-party notices.
The registration transform reader retains the MIT notice by Chengyue Wu.
StageWorld reference methods and component terms are documented in its retained license audit.
This private source collection does not grant additional rights to third-party weights or datasets.
No new blanket license is assigned to the collection. For project-specific details, use only rows present in the table above.

`SOURCE_FILES.json` maps copied source files. Newly written release tooling and documentation are recorded in `RELEASE_FILES.json`.
Source identity is described by project, branch/context and snapshot date; release records do not contain checksum values.
