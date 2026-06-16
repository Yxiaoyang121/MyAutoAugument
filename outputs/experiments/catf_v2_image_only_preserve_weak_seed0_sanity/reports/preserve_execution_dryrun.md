# Preserve Original Execution Dry-run

- Dry-run only, no training: `true`
- Expected union: `[4, 11, 12]`
- Runtime policy_matrix union: `[4, 11, 12]`
- Router eligible union: `[4, 11, 12]`
- Final executable union: `[4, 11, 12]`
- Runtime contains class4/11/12: `true`
- Router contains class4/11/12: `true`
- Fixed op list inherited: `true`
- Fixed prob/strength inherited: `true`
- Weak class9 replaced preserve path: `false`
- sampler_only: `false`
- weighted_index_list: `false`

| epoch | expected | runtime policy | router eligible | executable | runtime ops |
| --- | --- | --- | --- | --- | --- |
| 5 | `[4, 11]` | `[4, 11]` | `[4, 11]` | `[4, 11]` | `c4:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; c11:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02` |
| 10 | `[4]` | `[4]` | `[4]` | `[4]` | `c4:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01` |
| 15 | `[4, 12]` | `[4, 12]` | `[4, 12]` | `[4, 12]` | `c4:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; c12:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02` |
| 20 | `[4, 12]` | `[4, 12]` | `[4, 12]` | `[4, 12]` | `c4:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; c12:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02` |
| 25 | `[4, 12]` | `[4, 12]` | `[4, 12]` | `[4, 12]` | `c4:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; c12:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02` |
| 30 | `[4, 12]` | `[4, 12]` | `[4, 12]` | `[4, 12]` | `c4:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; c12:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02` |
| 35 | `[4, 12]` | `[4, 12]` | `[4, 12]` | `[4, 12]` | `c4:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; c12:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02` |
| 40 | `[4, 12]` | `[4, 12]` | `[4, 12]` | `[4, 12]` | `c4:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; c12:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02` |
| 45 | `[4, 12]` | `[4, 12]` | `[4, 12]` | `[4, 12]` | `c4:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; c12:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02` |
