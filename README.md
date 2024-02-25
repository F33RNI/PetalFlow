# 🌸 PetalFlow

| <img src="logo.png" height="100" height="100" alt="PetalFlow logo"> | <h3>Pure C library for machine learning</h3> |
| ------------------------------------------------------------------- | :------------------------------------------: |

----------

## ⌛ README in progress

For now, you can build tests by running

```shell
gcc -o petalflow test/main.c src/*.c -Iinclude -DLOGGING -DLOGGER_LEVEL=0 -lm
```

<details>
<summary>Output</summary>

```text

Testing activation functions on data:   -2.0000 -1.0000 0.0000  1.0000  2.0000

Linear activation with a=0.50, c=1.00:  0.0000  0.5000  1.0000  1.5000  2.0000
Derivative:                             0.5000  0.5000  0.5000  0.5000  0.5000
Derivative approximation:               0.5000  0.5000  0.5000  0.5000  0.5000
Passed

ReLU activation with leak=0.10:         -0.2000 -0.1000 0.0000  1.0000  2.0000
Derivative:                             0.1000  0.1000  1.0000  1.0000  1.0000
Derivative approximation:               0.1000  0.1000  1.0000  1.0000  0.9999
Passed

ELU activation with alpha=0.10:         -0.0865 -0.0632 0.0000  1.0000  2.0000
Derivative:                             0.0135  0.0368  1.0000  1.0000  1.0000
Derivative approximation:               0.0135  0.0368  1.0000  1.0000  0.9999
Passed

Softsign activation:                    -0.6667 -0.5000 0.0000  0.5000  0.6667
Derivative:                             0.1112  0.2503  0.9980  0.2498  0.1110
Derivative approximation:               0.1112  0.2501  0.9990  0.2499  0.1110
Passed

Sigmoid activation:                     0.1192  0.2689  0.5000  0.7311  0.8808
Derivative:                             0.1050  0.1966  0.2500  0.1966  0.1050
Derivative approximation:               0.1050  0.1966  0.2500  0.1965  0.1050
Passed

Hard-sigmoid activation:                0.1000  0.3000  0.5000  0.7000  0.9000
Derivative:                             0.2000  0.2000  0.2000  0.2000  0.2000
Derivative approximation:               0.2000  0.2000  0.2000  0.2000  0.2000
Passed

E-Swish activation with beta=2.00:      -0.4768 -0.5379 0.0000  1.4621  3.5232
Derivative:                             -0.1813 0.1452  1.0005  1.8554  2.1814
Derivative approximation:               -0.1816 0.1450  1.0005  1.8556  2.1818
Passed

Softmax activation:                     0.0117  0.0317  0.0861  0.2341  0.6364
Derivative:
0.0115  -0.0004 -0.0010 -0.0027 -0.0074
-0.0004 0.0307  -0.0027 -0.0074 -0.0202
-0.0010 -0.0027 0.0787  -0.0202 -0.0548
-0.0027 -0.0074 -0.0202 0.1793  -0.1490
-0.0074 -0.0202 -0.0548 -0.1490 0.2314
Passed

tanh activation:                        -0.9640 -0.7616 0.0000  0.7616  0.9640
Derivative:                             0.0707  0.4200  1.0000  0.4200  0.0707
Derivative approximation:               0.0708  0.4204  1.0000  0.4196  0.0706
Passed

[2024-02-25 15:09:53] [INFO] [activation_destroy] Destroying activation struct with address: 0x55c3c72f66b0
--------------------------------------------------------------------------------

Testing loss functions on predicted data:       0.0000  0.5000  0.1000  0.9000  0.4000  0.9000
Testing loss functions on expected data:        0.0000  0.0000  0.0000  1.0000  0.0000  0.0000

Mean squared loss:                              0.2067
Derivative:                                     -0.0000 0.1667  0.0333  -0.0333 0.1333  0.3000
Derivative approximation:                       0.0000  0.1667  0.0334  -0.0332 0.1334  0.3001
Passed

Mean squared logarithmic loss:                  0.1169
Derivative:                                     -0.0000 0.0901  0.0289  -0.0090 0.0801  0.1126
Derivative approximation:                       0.0000  0.0901  0.0289  -0.0090 0.0801  0.1126
Passed

Root mean squared logarithmic loss:             0.3419
Derivative:                                     -0.0000 0.1318  0.0422  -0.0132 0.1172  0.1647
Derivative approximation:                       0.0000  0.1318  0.0423  -0.0131 0.1172  0.1646
Passed

Mean absolute loss:                             0.3333
Derivative:                                     -0.0000 0.1667  0.1667  -0.1667 0.1667  0.1667
Derivative approximation:                       0.0000  0.1667  0.1669  -0.1667 0.1667  0.1667
Passed

Binary cross-entropy loss:                      0.6195
Derivative:                                     0.0000  0.3333  0.1852  -0.1852 0.2778  1.6667
Derivative approximation:                       0.0000  0.3335  0.1848  -0.1851 0.2779  1.6741
Passed

Categorical cross-entropy loss:                 0.1054
Derivative:                                     -0.0000 -0.0000 -0.0000 -1.1111 -0.0000 -0.0000
Derivative approximation:                       0.0000  0.0000  0.0000  -1.1105 0.0000  0.0000
Passed

[2024-02-25 15:09:53] [INFO] [loss_destroy] Destroying loss struct with address: 0x55c3c72f69d0
--------------------------------------------------------------------------------

Testing dropout on array with size 50 and ratio: 0.20
[2024-02-25 15:09:53] [INFO] [bit_array_init] Initializing bit array with size: 50 bits
Array of bits: 00000000000010010000010000010000010110000011000001
Bits set: 10 (20.0000%)
[2024-02-25 15:09:53] [INFO] [bit_array_destroy] Destroying bit array struct with address: 0x55c3c72f6aa0
Passed

--------------------------------------------------------------------------------

Testing normalization petals
[2024-02-25 15:09:53] [INFO] [petal_init] Initializing petal with type: 1
1D (PETAL_TYPE_NORMALIZE_ALL) Input data:
2.0000  0.0000  10.0000 -1.0000 1.0000  8.0000  2.0000  1.5000  0.5000  -0.4000 -0.1000 0.1000
Normalized:
-0.4545 -0.8182 1.0000  -1.0000 -0.6364 0.6364  -0.4545 -0.5455 -0.7273 -0.8909 -0.8364 -0.8000
Output range: -1.0000 to 1.0000
Passed
[2024-02-25 15:09:53] [INFO] [petal_destroy] Destroying petal struct with address: 0x55c3c72f6ae0

[2024-02-25 15:09:53] [INFO] [petal_init] Initializing petal with type: 2
2D (PETAL_TYPE_NORMALIZE_IN_ROWS) Input data:
2.0000  0.0000  10.0000 -1.0000
1.0000  8.0000  2.0000  1.5000
0.5000  -0.4000 -0.1000 0.1000
Normalized:
-0.4545 -0.8182 1.0000  -1.0000
-1.0000 1.0000  -0.7143 -0.8571
1.0000  -1.0000 -0.3333 0.1111
Output range: -1.0000 to 1.0000
Passed
[2024-02-25 15:09:53] [INFO] [petal_destroy] Destroying petal struct with address: 0x55c3c72f6bd0

[2024-02-25 15:09:53] [INFO] [petal_init] Initializing petal with type: 3
3D (PETAL_TYPE_NORMALIZE_IN_CHANNELS) Input data:
(2.0000, 0.0000)        (10.0000, -1.0000)
(1.0000, 8.0000)        (2.0000, 1.5000)
(0.5000, -0.4000)       (-0.1000, 0.1000)
Normalized:
(-0.5842, -0.7778)      (1.0000, -1.0000)
(-0.7822, 1.0000)       (-0.5842, -0.4444)
(-0.8812, -0.8667)      (-1.0000, -0.7556)
Output range: -1.0000 to 1.0000
Passed
[2024-02-25 15:09:53] [INFO] [petal_destroy] Destroying petal struct with address: 0x55c3c72f6cc0

--------------------------------------------------------------------------------

Testing simple classifier using 3 dense layers
[2024-02-25 15:09:53] [INFO] [petal_init] Initializing petal with type: 4
[2024-02-25 15:09:53] [INFO] [weights_init] Initializing weights using 4 initializer
[2024-02-25 15:09:53] [INFO] [weights_init] Initializing weights using 0 initializer
[2024-02-25 15:09:53] [INFO] [petal_init] Initializing petal with type: 4
[2024-02-25 15:09:53] [INFO] [weights_init] Initializing weights using 4 initializer
[2024-02-25 15:09:53] [INFO] [weights_init] Initializing weights using 0 initializer
[2024-02-25 15:09:53] [INFO] [petal_init] Initializing petal with type: 4
[2024-02-25 15:09:53] [INFO] [weights_init] Initializing weights using 4 initializer
[2024-02-25 15:09:53] [INFO] [weights_init] Initializing weights using 0 initializer
In -> hidden 1 weights:
0.7836  -1.4375
-2.6656 0.2539
In -> hidden 1 bias weights:
0.0000  0.0000
hidden 1 -> hidden 2 weights:
-1.0741 -2.2841
0.2285  0.2800
hidden 1 -> hidden 2 bias weights:
0.0000  0.0000
hidden 2 -> out weights:
0.3229  0.0141  -0.5694
-0.8443 -1.9946 -0.8646
hidden 2 -> out bias weights:
0.0000  0.0000  0.0000
[2024-02-25 15:09:53] [INFO] [flower_init] Initializing flower with 3 petals
Before training [1.0, 2.0] -> [1 > 2, 1 == 2, 1 < 2]:           0.3333  0.3333  0.3333
[2024-02-25 15:09:53] [INFO] [metrics_init] Initializing metrics with log_interval: 1
[2024-02-25 15:09:53] [INFO] [metrics_add] Added metric: 0
[2024-02-25 15:09:53] [INFO] [metrics_add] Added metric: 1
[2024-02-25 15:09:53] [INFO] [metrics_add] Added metric: 2
[2024-02-25 15:09:53] [INFO] [metrics_add] Added metric: 3
[2024-02-25 15:09:53] [INFO] [metrics_add] Added metric: 4
[2024-02-25 15:09:53] [INFO] [flower_train] Training started
[2024-02-25 15:09:53] [INFO] [flower_train] Epoch: 1/10
[2024-02-25 15:09:53] [INFO] [Metrics] E: 1/10, B: 8/8, Time since start: 00:00:00 | since start of epoch: 00:00:00
[2024-02-25 15:09:53] [INFO] [Metrics] E: 1/10, B: 8/8, Train loss: 0.638303 (current) | 0.811049 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 1/10, B: 8/8, Train accuracy: 78.6667% (current) | 75.3333% (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 1/10, B: 8/8, Validation loss: 0.599726 (current) | 0.828731 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 1/10, B: 8/8, Validation accuracy: 84.6667% (current) | 75.9167% (epoch avg)
[2024-02-25 15:09:53] [INFO] [flower_train] Epoch: 2/10
[2024-02-25 15:09:53] [INFO] [Metrics] E: 2/10, B: 8/8, Time since start: 00:00:00 | since start of epoch: 00:00:00
[2024-02-25 15:09:53] [INFO] [Metrics] E: 2/10, B: 8/8, Train loss: 0.451966 (current) | 0.454896 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 2/10, B: 8/8, Train accuracy: 89.3333% (current) | 91.0000% (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 2/10, B: 8/8, Validation loss: 0.347247 (current) | 0.426023 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 2/10, B: 8/8, Validation accuracy: 92.6667% (current) | 92.1667% (epoch avg)
[2024-02-25 15:09:53] [INFO] [flower_train] Epoch: 3/10
[2024-02-25 15:09:53] [INFO] [Metrics] E: 3/10, B: 8/8, Time since start: 00:00:00 | since start of epoch: 00:00:00
[2024-02-25 15:09:53] [INFO] [Metrics] E: 3/10, B: 8/8, Train loss: 0.387405 (current) | 0.336819 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 3/10, B: 8/8, Train accuracy: 89.3333% (current) | 91.8333% (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 3/10, B: 8/8, Validation loss: 0.287836 (current) | 0.310859 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 3/10, B: 8/8, Validation accuracy: 92.6667% (current) | 92.6667% (epoch avg)
[2024-02-25 15:09:53] [INFO] [flower_train] Epoch: 4/10
[2024-02-25 15:09:53] [INFO] [Metrics] E: 4/10, B: 8/8, Time since start: 00:00:00 | since start of epoch: 00:00:00
[2024-02-25 15:09:53] [INFO] [Metrics] E: 4/10, B: 8/8, Train loss: 0.268021 (current) | 0.269400 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 4/10, B: 8/8, Train accuracy: 96.0000% (current) | 94.1667% (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 4/10, B: 8/8, Validation loss: 0.196923 (current) | 0.243808 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 4/10, B: 8/8, Validation accuracy: 98.6667% (current) | 95.1667% (epoch avg)
[2024-02-25 15:09:53] [INFO] [flower_train] Epoch: 5/10
[2024-02-25 15:09:53] [INFO] [Metrics] E: 5/10, B: 8/8, Time since start: 00:00:00 | since start of epoch: 00:00:00
[2024-02-25 15:09:53] [INFO] [Metrics] E: 5/10, B: 8/8, Train loss: 0.165997 (current) | 0.164480 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 5/10, B: 8/8, Train accuracy: 97.3333% (current) | 98.3333% (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 5/10, B: 8/8, Validation loss: 0.099338 (current) | 0.147046 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 5/10, B: 8/8, Validation accuracy: 100.0000% (current) | 99.7500% (epoch avg)
[2024-02-25 15:09:53] [INFO] [flower_train] Epoch: 6/10
[2024-02-25 15:09:53] [INFO] [Metrics] E: 6/10, B: 8/8, Time since start: 00:00:00 | since start of epoch: 00:00:00
[2024-02-25 15:09:53] [INFO] [Metrics] E: 6/10, B: 8/8, Train loss: 0.087892 (current) | 0.085062 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 6/10, B: 8/8, Train accuracy: 100.0000% (current) | 99.5000% (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 6/10, B: 8/8, Validation loss: 0.081233 (current) | 0.087904 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 6/10, B: 8/8, Validation accuracy: 100.0000% (current) | 99.5833% (epoch avg)
[2024-02-25 15:09:53] [INFO] [flower_train] Epoch: 7/10
[2024-02-25 15:09:53] [INFO] [Metrics] E: 7/10, B: 8/8, Time since start: 00:00:00 | since start of epoch: 00:00:00
[2024-02-25 15:09:53] [INFO] [Metrics] E: 7/10, B: 8/8, Train loss: 0.046620 (current) | 0.051340 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 7/10, B: 8/8, Train accuracy: 100.0000% (current) | 100.0000% (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 7/10, B: 8/8, Validation loss: 0.039401 (current) | 0.046568 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 7/10, B: 8/8, Validation accuracy: 100.0000% (current) | 100.0000% (epoch avg)
[2024-02-25 15:09:53] [INFO] [flower_train] Epoch: 8/10
[2024-02-25 15:09:53] [INFO] [Metrics] E: 8/10, B: 8/8, Time since start: 00:00:00 | since start of epoch: 00:00:00
[2024-02-25 15:09:53] [INFO] [Metrics] E: 8/10, B: 8/8, Train loss: 0.036631 (current) | 0.034752 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 8/10, B: 8/8, Train accuracy: 100.0000% (current) | 100.0000% (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 8/10, B: 8/8, Validation loss: 0.030337 (current) | 0.038632 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 8/10, B: 8/8, Validation accuracy: 100.0000% (current) | 100.0000% (epoch avg)
[2024-02-25 15:09:53] [INFO] [flower_train] Epoch: 9/10
[2024-02-25 15:09:53] [INFO] [Metrics] E: 9/10, B: 8/8, Time since start: 00:00:00 | since start of epoch: 00:00:00
[2024-02-25 15:09:53] [INFO] [Metrics] E: 9/10, B: 8/8, Train loss: 0.030700 (current) | 0.027104 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 9/10, B: 8/8, Train accuracy: 100.0000% (current) | 100.0000% (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 9/10, B: 8/8, Validation loss: 0.029901 (current) | 0.028329 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 9/10, B: 8/8, Validation accuracy: 100.0000% (current) | 100.0000% (epoch avg)
[2024-02-25 15:09:53] [INFO] [flower_train] Epoch: 10/10
[2024-02-25 15:09:53] [INFO] [Metrics] E: 10/10, B: 8/8, Time since start: 00:00:00 | since start of epoch: 00:00:00
[2024-02-25 15:09:53] [INFO] [Metrics] E: 10/10, B: 8/8, Train loss: 0.024623 (current) | 0.021676 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 10/10, B: 8/8, Train accuracy: 100.0000% (current) | 100.0000% (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 10/10, B: 8/8, Validation loss: 0.020332 (current) | 0.023964 (epoch avg)
[2024-02-25 15:09:53] [INFO] [Metrics] E: 10/10, B: 8/8, Validation accuracy: 100.0000% (current) | 100.0000% (epoch avg)
After training [1.0, 20.0] -> [1 > 2, 1 == 2, 1 < 2]:           0.0001  0.0135  0.9864
After training [5.0, 5.0] -> [1 > 2, 1 == 2, 1 < 2]:            0.0330  0.8270  0.1400
After training [-1.0, -100.0] -> [1 > 2, 1 == 2, 1 < 2]:        1.0000  0.0000  0.0000
Min flower size: 1348 bytes
[2024-02-25 15:09:53] [INFO] [weights_destroy] Destroying weights struct's (address: 0x7ffd97b165f0) internal data
[2024-02-25 15:09:53] [INFO] [weights_destroy] Destroying weights struct's (address: 0x7ffd97b16630) internal data
[2024-02-25 15:09:53] [INFO] [weights_destroy] Destroying weights struct's (address: 0x7ffd97b16670) internal data
[2024-02-25 15:09:53] [INFO] [weights_destroy] Destroying weights struct's (address: 0x7ffd97b166b0) internal data
[2024-02-25 15:09:53] [INFO] [weights_destroy] Destroying weights struct's (address: 0x7ffd97b166f0) internal data
[2024-02-25 15:09:53] [INFO] [weights_destroy] Destroying weights struct's (address: 0x7ffd97b16730) internal data
[2024-02-25 15:09:53] [INFO] [flower_destroy] Destroying flower struct with address: 0x55c3c7300d30
[2024-02-25 15:09:53] [INFO] [loss_destroy] Destroying loss struct with address: 0x55c3c7300e40
[2024-02-25 15:09:53] [INFO] [metrics_destroy] Destroying metrics struct with address: 0x55c3c7300dc0
--------------------------------------------------------------------------------

Fails: 0
All tests passed successfully!
```

</details>

----------

## 🚧 Petal-Flow is under development

Only alpha version available now
