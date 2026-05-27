# YOLO Default Feedback vs References

- Reference for constraints: YOLO default seed=42.
- Constraint rule: fail if Precision, mAP50, or mAP50-95 drops by more than 0.01.

| reference | P | R | mAP50 | mAP50-95 | dP | dR | d_mAP50 | d_mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO default seed=42 | 0.7132 | 0.7600 | 0.7759 | 0.5241 | +0.0580 | -0.0911 | -0.0320 | -0.0248 |
| baseline no aug | 0.6900 | 0.6150 | 0.6690 | 0.4340 | +0.0812 | +0.0539 | +0.0749 | +0.0653 |
| offline DiagAug | 0.6860 | 0.6880 | 0.7170 | 0.4960 | +0.0852 | -0.0191 | +0.0269 | +0.0033 |
| offline random external | 0.7500 | 0.6680 | 0.7340 | 0.5010 | +0.0212 | +0.0009 | +0.0099 | -0.0017 |

## Conclusion

- Constraint failed: `true`
- Final strategy accepted: `false`
