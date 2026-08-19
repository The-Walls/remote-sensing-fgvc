# Experiment ladder

baseline = `L0`; Δ columns are absolute percentage points against it. `± ` appears once a rung has >1 seed.

| id | backbone | img | aug | head | loss | seeds | overall top-1 | Δ | mean-per-class | Δ | MPC (support≥5) | macro-F1 | params (M) | s/epoch | Δ vs parent |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L0 | resnet50 | 224 | basic | gap | ce | 1 | 98.62 | +0.00 | 90.27 | +0.00 | 98.04 | 89.88 | 23.59 | 7.9 | — |
| L0_naive_split | resnet50 | 224 | basic | gap | ce | 1 | 98.46 | -0.16 | 92.23 | +1.96 | 98.29 | 91.60 | 23.59 | 7.3 | `data.group_aware_split`: True→False |
| L1_320 | resnet50 | 320 | basic | gap | ce | 1 | 99.31 | +0.69 | 92.69 | +2.42 | 98.09 | 93.19 | 23.59 | 14.0 | `data.img_size`: 224→320 |
| L1_448 | resnet50 | 448 | basic | gap | ce | 1 | 99.10 | +0.48 | 92.81 | +2.54 | 98.27 | 92.57 | 23.59 | 26.0 | `data.img_size`: 224→448 |
| L2 | resnet50 | 448 | rs_rot | gap | ce | 1 | 99.79 | +1.17 | 97.46 | +7.19 | 99.78 | 97.31 | 23.59 | 26.0 | `data.aug`: basic→rs_rot |
| L3_compact_bilinear | resnet50 | 448 | rs_rot | compact_bilinear | ce | 1 | 99.15 | +0.53 | 91.62 | +1.35 | 98.27 | 91.27 | 23.85 | 31.1 | `model.head`: gap→compact_bilinear |
| L4_cbam | resnet50 | 448 | rs_rot | cbam | ce | 1 | 99.73 | +1.12 | 96.27 | +6.00 | 99.78 | 96.50 | 24.12 | 27.3 | `model.head`: gap→cbam |
| L5_cb_ce | resnet50 | 448 | rs_rot | gap | cb_ce | 1 | 99.89 | +1.28 | 97.49 | +7.22 | 99.82 | 97.37 | 23.59 | 23.1 | `loss.name`: ce→cb_ce |
| L6a_convnext_tiny | convnext_tiny | 448 | rs_rot | gap | ce | 1 | 99.89 | +1.28 | 97.49 | +7.22 | 99.82 | 97.35 | 27.85 | 23.4 | `model.name`: resnet50→convnext_tiny |
| L6b_vit_base_p16 | vit_base_patch16_224 | 448 | rs_rot | gap | ce | 1 | 99.89 | +1.28 | 97.49 | +7.22 | 99.82 | 96.58 | 86.28 | 50.4 | `model.name`: resnet50→vit_base_patch16_224 |

## Most confusable class pairs (top-15 per run)

### L0

| true | predicted | n | true support |
|---|---|---:|---:|
| 026.Cargo_ship | 033.Tank_ship | 5 | 90 |
| 018.Whitby_Island-class_dock_landing_ship | 009.Arleigh_Burke-class_destroyer | 3 | 66 |
| 008.Ticonderoga-class_cruiser | 014.Wasp-class_assault_ship | 2 | 150 |
| 037.Horizon-class_destroyer | 019.San_Antonio-class_transport_dock | 1 | 1 |
| 039.Mistral-class_amphibious_assault_ship | 023.Crane_ship | 1 | 1 |
| 019.San_Antonio-class_transport_dock | 018.Whitby_Island-class_dock_landing_ship | 1 | 74 |
| 018.Whitby_Island-class_dock_landing_ship | 020.Freedom-class_combat_ship | 1 | 66 |
| 018.Whitby_Island-class_dock_landing_ship | 019.San_Antonio-class_transport_dock | 1 | 66 |
| 018.Whitby_Island-class_dock_landing_ship | 008.Ticonderoga-class_cruiser | 1 | 66 |
| 011.Asagiri-class_destroyer | 038.Atago-class_destroyer | 1 | 19 |
| 010.Akizuki-class_destroyer | 038.Atago-class_destroyer | 1 | 6 |
| 010.Akizuki-class_destroyer | 008.Ticonderoga-class_cruiser | 1 | 6 |
| 009.Arleigh_Burke-class_destroyer | 018.Whitby_Island-class_dock_landing_ship | 1 | 143 |
| 009.Arleigh_Burke-class_destroyer | 008.Ticonderoga-class_cruiser | 1 | 143 |
| 008.Ticonderoga-class_cruiser | 018.Whitby_Island-class_dock_landing_ship | 1 | 150 |

![errors](errors_L0.png)

### L0_naive_split

| true | predicted | n | true support |
|---|---|---:|---:|
| 008.Ticonderoga-class_cruiser | 009.Arleigh_Burke-class_destroyer | 4 | 152 |
| 011.Asagiri-class_destroyer | 016.Hyuga-class_helicopter_destroyer | 3 | 18 |
| 009.Arleigh_Burke-class_destroyer | 018.Whitby_Island-class_dock_landing_ship | 3 | 145 |
| 009.Arleigh_Burke-class_destroyer | 038.Atago-class_destroyer | 2 | 145 |
| 009.Arleigh_Burke-class_destroyer | 008.Ticonderoga-class_cruiser | 2 | 145 |
| 026.Cargo_ship | 033.Tank_ship | 1 | 94 |
| 037.Horizon-class_destroyer | 008.Ticonderoga-class_cruiser | 1 | 1 |
| 027.Murasame-class_destroyer | 011.Asagiri-class_destroyer | 1 | 16 |
| 039.Mistral-class_amphibious_assault_ship | 025.Megayacht | 1 | 1 |
| 019.San_Antonio-class_transport_dock | 018.Whitby_Island-class_dock_landing_ship | 1 | 80 |
| 018.Whitby_Island-class_dock_landing_ship | 019.San_Antonio-class_transport_dock | 1 | 70 |
| 018.Whitby_Island-class_dock_landing_ship | 009.Arleigh_Burke-class_destroyer | 1 | 70 |
| 018.Whitby_Island-class_dock_landing_ship | 008.Ticonderoga-class_cruiser | 1 | 70 |
| 014.Wasp-class_assault_ship | 001.Nimitz-class_aircraft_carrier | 1 | 113 |
| 011.Asagiri-class_destroyer | 027.Murasame-class_destroyer | 1 | 18 |

![errors](errors_L0_naive_split.png)

### L1_320

| true | predicted | n | true support |
|---|---|---:|---:|
| 026.Cargo_ship | 033.Tank_ship | 2 | 90 |
| 010.Akizuki-class_destroyer | 038.Atago-class_destroyer | 2 | 6 |
| 035.Zumwalt-class_destroyer | 019.San_Antonio-class_transport_dock | 1 | 2 |
| 022.Sacramento-class_support_ship | 008.Ticonderoga-class_cruiser | 1 | 8 |
| 032.Sand_carrier | 026.Cargo_ship | 1 | 56 |
| 037.Horizon-class_destroyer | 009.Arleigh_Burke-class_destroyer | 1 | 1 |
| 019.San_Antonio-class_transport_dock | 018.Whitby_Island-class_dock_landing_ship | 1 | 74 |
| 011.Asagiri-class_destroyer | 038.Atago-class_destroyer | 1 | 19 |
| 009.Arleigh_Burke-class_destroyer | 008.Ticonderoga-class_cruiser | 1 | 143 |
| 007.INS_Vikramaditya_aircraft_carrier | 006.INS_Virrat_aircraft_carrier | 1 | 2 |
| 005.Charles_de_Gaulle_aricraft_carrier | 007.INS_Vikramaditya_aircraft_carrier | 1 | 2 |

![errors](errors_L1_320.png)

### L1_448

| true | predicted | n | true support |
|---|---|---:|---:|
| 009.Arleigh_Burke-class_destroyer | 018.Whitby_Island-class_dock_landing_ship | 3 | 143 |
| 026.Cargo_ship | 033.Tank_ship | 2 | 90 |
| 010.Akizuki-class_destroyer | 038.Atago-class_destroyer | 2 | 6 |
| 008.Ticonderoga-class_cruiser | 009.Arleigh_Burke-class_destroyer | 2 | 150 |
| 032.Sand_carrier | 025.Megayacht | 1 | 56 |
| 026.Cargo_ship | 030.Civil_yacht | 1 | 90 |
| 023.Crane_ship | 001.Nimitz-class_aircraft_carrier | 1 | 33 |
| 032.Sand_carrier | 026.Cargo_ship | 1 | 56 |
| 039.Mistral-class_amphibious_assault_ship | 009.Arleigh_Burke-class_destroyer | 1 | 1 |
| 037.Horizon-class_destroyer | 026.Cargo_ship | 1 | 1 |
| 011.Asagiri-class_destroyer | 038.Atago-class_destroyer | 1 | 19 |
| 005.Charles_de_Gaulle_aricraft_carrier | 007.INS_Vikramaditya_aircraft_carrier | 1 | 2 |

![errors](errors_L1_448.png)

### L2

| true | predicted | n | true support |
|---|---|---:|---:|
| 037.Horizon-class_destroyer | 026.Cargo_ship | 1 | 1 |
| 011.Asagiri-class_destroyer | 038.Atago-class_destroyer | 1 | 19 |
| 009.Arleigh_Burke-class_destroyer | 018.Whitby_Island-class_dock_landing_ship | 1 | 143 |
| 008.Ticonderoga-class_cruiser | 018.Whitby_Island-class_dock_landing_ship | 1 | 150 |

![errors](errors_L2.png)

### L3_compact_bilinear

| true | predicted | n | true support |
|---|---|---:|---:|
| 023.Crane_ship | 001.Nimitz-class_aircraft_carrier | 3 | 33 |
| 008.Ticonderoga-class_cruiser | 014.Wasp-class_assault_ship | 3 | 150 |
| 010.Akizuki-class_destroyer | 038.Atago-class_destroyer | 2 | 6 |
| 009.Arleigh_Burke-class_destroyer | 018.Whitby_Island-class_dock_landing_ship | 2 | 143 |
| 007.INS_Vikramaditya_aircraft_carrier | 001.Nimitz-class_aircraft_carrier | 2 | 2 |
| 039.Mistral-class_amphibious_assault_ship | 006.INS_Virrat_aircraft_carrier | 1 | 1 |
| 037.Horizon-class_destroyer | 026.Cargo_ship | 1 | 1 |
| 011.Asagiri-class_destroyer | 038.Atago-class_destroyer | 1 | 19 |
| 008.Ticonderoga-class_cruiser | 018.Whitby_Island-class_dock_landing_ship | 1 | 150 |

![errors](errors_L3_compact_bilinear.png)

### L4_cbam

| true | predicted | n | true support |
|---|---|---:|---:|
| 037.Horizon-class_destroyer | 021.Independence-class_combat_ship | 1 | 1 |
| 011.Asagiri-class_destroyer | 038.Atago-class_destroyer | 1 | 19 |
| 008.Ticonderoga-class_cruiser | 018.Whitby_Island-class_dock_landing_ship | 1 | 150 |
| 008.Ticonderoga-class_cruiser | 001.Nimitz-class_aircraft_carrier | 1 | 150 |
| 007.INS_Vikramaditya_aircraft_carrier | 001.Nimitz-class_aircraft_carrier | 1 | 2 |

![errors](errors_L4_cbam.png)

### L5_cb_ce

| true | predicted | n | true support |
|---|---|---:|---:|
| 037.Horizon-class_destroyer | 030.Civil_yacht | 1 | 1 |
| 011.Asagiri-class_destroyer | 038.Atago-class_destroyer | 1 | 19 |

![errors](errors_L5_cb_ce.png)

### L6a_convnext_tiny

| true | predicted | n | true support |
|---|---|---:|---:|
| 037.Horizon-class_destroyer | 032.Sand_carrier | 1 | 1 |
| 011.Asagiri-class_destroyer | 038.Atago-class_destroyer | 1 | 19 |

![errors](errors_L6a_convnext_tiny.png)

### L6b_vit_base_p16

| true | predicted | n | true support |
|---|---|---:|---:|
| 037.Horizon-class_destroyer | 039.Mistral-class_amphibious_assault_ship | 1 | 1 |
| 011.Asagiri-class_destroyer | 038.Atago-class_destroyer | 1 | 19 |

![errors](errors_L6b_vit_base_p16.png)
