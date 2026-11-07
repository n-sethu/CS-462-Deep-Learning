# Backup presentation

The recording is an actual browser capture of the same local Streamlit app, using fixed ensemble 01 at 90% coverage. It is silent; use the narration in the README. Counts are specific to this ensemble and must not be presented as aggregate ten-ensemble effects.

| Condition | Training | A errors | B errors | A AURC | B AURC |
|---|---|---:|---:|---:|---:|
| clean | clean_trained | 140 | 133 | 0.005544 | 0.005022 |
| gaussian_noise_3 | clean_trained | 5235 | 5229 | 0.422754 | 0.427671 |
| gaussian_noise_3 | adapted | 5216 | 5188 | 0.419832 | 0.418096 |
| motion_blur_3 | adapted | 1837 | 1886 | 0.085318 | 0.084664 |

Important: in this fixed example ensemble, clean-trained B catches six more errors at exactly 90% under noise 3 while having worse integrated AURC. Use the curve/AURC and replication tab to explain transfer failure; do not claim all noise operating points worsen.
