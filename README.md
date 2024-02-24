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

[2024-02-24 15:21:08] [INFO] [activation_destroy] Destroying activation struct with address: 0x55b2aca636b0
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

[2024-02-24 15:21:08] [INFO] [loss_destroy] Destroying loss struct with address: 0x55b2aca639d0
--------------------------------------------------------------------------------

Testing dropout on array with size 50 and ratio: 0.20
[2024-02-24 15:21:08] [INFO] [bit_array_init] Initializing bit array with size: 50 bits
Array of bits: 00000000000010010000010000010000010110000011000001
Bits set: 10 (20.0000%)
[2024-02-24 15:21:08] [INFO] [bit_array_destroy] Destroying bit array struct with address: 0x55b2aca63aa0
Passed

--------------------------------------------------------------------------------

Testing normalization petals
[2024-02-24 15:21:08] [INFO] [petal_init] Initializing petal with type: 1
1D (PETAL_TYPE_NORMALIZE_ALL) Input data:
2.0000  0.0000  10.0000 -1.0000 1.0000  8.0000  2.0000  1.5000  0.5000  -0.4000 -0.1000 0.1000
Normalized:
-0.4545 -0.8182 1.0000  -1.0000 -0.6364 0.6364  -0.4545 -0.5455 -0.7273 -0.8909 -0.8364 -0.8000
Output range: -1.0000 to 1.0000
Passed
[2024-02-24 15:21:08] [INFO] [petal_destroy] Destroying petal struct with address: 0x55b2aca63ae0

[2024-02-24 15:21:08] [INFO] [petal_init] Initializing petal with type: 2
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
[2024-02-24 15:21:08] [INFO] [petal_destroy] Destroying petal struct with address: 0x55b2aca63bd0

[2024-02-24 15:21:08] [INFO] [petal_init] Initializing petal with type: 3
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
[2024-02-24 15:21:08] [INFO] [petal_destroy] Destroying petal struct with address: 0x55b2aca63cc0

--------------------------------------------------------------------------------

Testing simple classifier using 3 dense layers
[2024-02-24 15:21:08] [INFO] [petal_init] Initializing petal with type: 4
[2024-02-24 15:21:08] [INFO] [weights_init] Initializing weights using 4 initializer
[2024-02-24 15:21:08] [INFO] [weights_init] Initializing weights using 0 initializer
[2024-02-24 15:21:08] [INFO] [petal_init] Initializing petal with type: 4
[2024-02-24 15:21:08] [INFO] [weights_init] Initializing weights using 4 initializer
[2024-02-24 15:21:08] [INFO] [weights_init] Initializing weights using 0 initializer
[2024-02-24 15:21:08] [INFO] [petal_init] Initializing petal with type: 4
[2024-02-24 15:21:08] [INFO] [weights_init] Initializing weights using 4 initializer
[2024-02-24 15:21:08] [INFO] [weights_init] Initializing weights using 0 initializer
In -> hidden 1 weights:
0.6161  2.8460
0.6274  -1.7371
In -> hidden 1 bias weights:
0.0000  0.0000
hidden 1 -> hidden 2 weights:
-1.5716 -0.0869
-0.6270 0.7464
hidden 1 -> hidden 2 bias weights:
0.0000  0.0000
hidden 2 -> out weights:
-1.3622 -0.9300 0.8285
-0.3411 1.0539  0.5326
hidden 2 -> out bias weights:
0.0000  0.0000  0.0000
[2024-02-24 15:21:08] [INFO] [flower_init] Initializing flower with 3 petals
Before training [1.0, 2.0] -> [1 > 2, 1 == 2, 1 < 2]:           0.3333  0.3333  0.3333
Epoch = 0
[Train] _loss:  1.098612, accuracy:  43.33% avg | [Test] _loss:  1.997866 avg, accuracy:  56.67% avg
Epoch = 1
[Train] _loss:  1.236036, accuracy:  54.17% avg | [Test] _loss:  1.476556 avg, accuracy:  58.33% avg
Epoch = 2
[Train] _loss:  1.396040, accuracy:  57.08% avg | [Test] _loss:  0.949210 avg, accuracy:  60.00% avg
Epoch = 3
[Train] _loss:  1.619377, accuracy:  60.42% avg | [Test] _loss:  0.854820 avg, accuracy:  61.67% avg
Epoch = 4
[Train] _loss:  1.865430, accuracy:  65.42% avg | [Test] _loss:  0.853722 avg, accuracy:  68.33% avg
Epoch = 5
[Train] _loss:  2.081839, accuracy:  72.50% avg | [Test] _loss:  0.798336 avg, accuracy:  68.33% avg
Epoch = 6
[Train] _loss:  2.271788, accuracy:  73.75% avg | [Test] _loss:  0.758320 avg, accuracy:  68.33% avg
Epoch = 7
[Train] _loss:  2.444878, accuracy:  73.75% avg | [Test] _loss:  0.663603 avg, accuracy:  68.33% avg
Epoch = 8
[Train] _loss:  2.602095, accuracy:  62.50% avg | [Test] _loss:  0.668422 avg, accuracy:  68.33% avg
Epoch = 9
[Train] _loss:  2.732192, accuracy:  62.50% avg | [Test] _loss:  0.666517 avg, accuracy:  68.33% avg
Epoch = 10
[Train] _loss:  2.815355, accuracy:  62.50% avg | [Test] _loss:  0.627327 avg, accuracy:  68.33% avg
Epoch = 11
[Train] _loss:  2.845786, accuracy:  62.50% avg | [Test] _loss:  0.591590 avg, accuracy:  68.33% avg
Epoch = 12
[Train] _loss:  2.832634, accuracy:  62.50% avg | [Test] _loss:  0.606988 avg, accuracy:  68.33% avg
Epoch = 13
[Train] _loss:  2.790113, accuracy:  62.50% avg | [Test] _loss:  0.560583 avg, accuracy:  80.00% avg
Epoch = 14
[Train] _loss:  2.733489, accuracy:  82.50% avg | [Test] _loss:  0.549176 avg, accuracy:  80.00% avg
Epoch = 15
[Train] _loss:  2.667285, accuracy:  82.92% avg | [Test] _loss:  0.581600 avg, accuracy:  80.00% avg
Epoch = 16
[Train] _loss:  2.596571, accuracy:  84.17% avg | [Test] _loss:  0.561822 avg, accuracy:  81.67% avg
Epoch = 17
[Train] _loss:  2.517557, accuracy:  84.58% avg | [Test] _loss:  0.515555 avg, accuracy:  83.33% avg
Epoch = 18
[Train] _loss:  2.422116, accuracy:  85.83% avg | [Test] _loss:  0.496421 avg, accuracy:  83.33% avg
Epoch = 19
[Train] _loss:  2.306513, accuracy:  85.83% avg | [Test] _loss:  0.512657 avg, accuracy:  83.33% avg
Epoch = 20
[Train] _loss:  2.179319, accuracy:  86.67% avg | [Test] _loss:  0.484845 avg, accuracy:  85.00% avg
Epoch = 21
[Train] _loss:  2.049450, accuracy:  87.92% avg | [Test] _loss:  0.439805 avg, accuracy:  86.67% avg
Epoch = 22
[Train] _loss:  1.931860, accuracy:  88.75% avg | [Test] _loss:  0.430609 avg, accuracy:  86.67% avg
Epoch = 23
[Train] _loss:  1.838300, accuracy:  88.75% avg | [Test] _loss:  0.428415 avg, accuracy:  88.33% avg
Epoch = 24
[Train] _loss:  1.761157, accuracy:  88.75% avg | [Test] _loss:  0.374058 avg, accuracy:  88.33% avg
Epoch = 25
[Train] _loss:  1.720810, accuracy:  88.75% avg | [Test] _loss:  0.333276 avg, accuracy:  91.67% avg
Epoch = 26
[Train] _loss:  1.712081, accuracy:  90.42% avg | [Test] _loss:  0.293336 avg, accuracy:  91.67% avg
Epoch = 27
[Train] _loss:  1.734871, accuracy:  90.83% avg | [Test] _loss:  0.219733 avg, accuracy:  93.33% avg
Epoch = 28
[Train] _loss:  1.791233, accuracy:  91.67% avg | [Test] _loss:  0.214253 avg, accuracy:  96.67% avg
Epoch = 29
[Train] _loss:  1.867307, accuracy:  92.50% avg | [Test] _loss:  0.188096 avg, accuracy:  96.67% avg
Epoch = 30
[Train] _loss:  1.948355, accuracy:  93.75% avg | [Test] _loss:  0.160551 avg, accuracy:  96.67% avg
Epoch = 31
[Train] _loss:  2.035618, accuracy:  94.58% avg | [Test] _loss:  0.177040 avg, accuracy:  96.67% avg
Epoch = 32
[Train] _loss:  2.693866, accuracy:  94.58% avg | [Test] _loss:  0.184271 avg, accuracy:  96.67% avg
Epoch = 33
[Train] _loss:  1.540466, accuracy:  94.58% avg | [Test] _loss:  0.175238 avg, accuracy:  96.67% avg
Epoch = 34
[Train] _loss:  2.297769, accuracy:  94.58% avg | [Test] _loss:  0.169933 avg, accuracy:  96.67% avg
Epoch = 35
[Train] _loss:  2.337718, accuracy:  94.17% avg | [Test] _loss:  0.136743 avg, accuracy:  96.67% avg
Epoch = 36
[Train] _loss:  2.350053, accuracy:  94.58% avg | [Test] _loss:  0.159082 avg, accuracy:  96.67% avg
Epoch = 37
[Train] _loss:  2.340817, accuracy:  94.58% avg | [Test] _loss:  0.202636 avg, accuracy:  95.00% avg
Epoch = 38
[Train] _loss:  2.280632, accuracy:  94.17% avg | [Test] _loss:  0.191693 avg, accuracy:  95.00% avg
Epoch = 39
[Train] _loss:  1.131709, accuracy:  93.75% avg | [Test] _loss:  0.128861 avg, accuracy:  98.33% avg
Epoch = 40
[Train] _loss:  0.680395, accuracy:  96.67% avg | [Test] _loss:  0.138205 avg, accuracy:  96.67% avg
Epoch = 41
[Train] _loss:  0.924980, accuracy:  95.42% avg | [Test] _loss:  0.109968 avg, accuracy:  98.33% avg
Epoch = 42
[Train] _loss:  0.525228, accuracy:  97.50% avg | [Test] _loss:  0.106712 avg, accuracy:  96.67% avg
Epoch = 43
[Train] _loss:  0.679223, accuracy:  97.08% avg | [Test] _loss:  0.082265 avg, accuracy:  98.33% avg
Epoch = 44
[Train] _loss:  0.678739, accuracy:  98.75% avg | [Test] _loss:  0.066340 avg, accuracy:  98.33% avg
Epoch = 45
[Train] _loss:  2.499421, accuracy:  96.25% avg | [Test] _loss:  0.128891 avg, accuracy:  96.67% avg
Epoch = 46
[Train] _loss:  1.586578, accuracy:  95.83% avg | [Test] _loss:  0.081027 avg, accuracy:  96.67% avg
Epoch = 47
[Train] _loss:  0.982025, accuracy:  95.42% avg | [Test] _loss:  0.059623 avg, accuracy:  98.33% avg
Epoch = 48
[Train] _loss:  1.416196, accuracy:  97.08% avg | [Test] _loss:  0.068073 avg, accuracy:  98.33% avg
Epoch = 49
[Train] _loss:  0.964430, accuracy:  97.08% avg | [Test] _loss:  0.079407 avg, accuracy: 100.00% avg
After training [1.0, 20.0] -> [1 > 2, 1 == 2, 1 < 2]:           0.0000  0.0518  0.9482
After training [5.0, 5.0] -> [1 > 2, 1 == 2, 1 < 2]:            0.0009  0.5774  0.4217
After training [-1.0, -100.0] -> [1 > 2, 1 == 2, 1 < 2]:        0.9452  0.0536  0.0012
Min flower size: 1348 bytes
[2024-02-24 15:21:09] [INFO] [weights_destroy] Destroying weights struct's (address: 0x7ffed36c8360) internal data
[2024-02-24 15:21:09] [INFO] [weights_destroy] Destroying weights struct's (address: 0x7ffed36c83a0) internal data
[2024-02-24 15:21:09] [INFO] [weights_destroy] Destroying weights struct's (address: 0x7ffed36c83e0) internal data
[2024-02-24 15:21:09] [INFO] [weights_destroy] Destroying weights struct's (address: 0x7ffed36c8420) internal data
[2024-02-24 15:21:09] [INFO] [weights_destroy] Destroying weights struct's (address: 0x7ffed36c8460) internal data
[2024-02-24 15:21:09] [INFO] [weights_destroy] Destroying weights struct's (address: 0x7ffed36c84a0) internal data
[2024-02-24 15:21:09] [INFO] [flower_destroy] Destroying flower struct with address: 0x55b2aca67f60
[2024-02-24 15:21:09] [INFO] [loss_destroy] Destroying loss struct with address: 0x55b2aca67ff0
--------------------------------------------------------------------------------

Fails: 0
All tests passed successfully!
```

</details>

----------

## 🚧 Petal-Flow is under development

Only alpha version available now
