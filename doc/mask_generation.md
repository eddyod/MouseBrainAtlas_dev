
- On a machine with monitor, launch the maskingGUI. Run `DATA_ROOTDIR=/home/yuncong/brainstem/home/yuncong/demo_data ROOT_DIR=/home/yuncong/brainstem/home/yuncong/demo_data THUMBNAIL_DATA_ROOTDIR=/home/yuncong/brainstem/home/yuncong/demo_data python src/gui/mask_editing_tool_v4.py DEMO998 NtbNormalized`. Generate initial masks.

```bash
├── CSHL_data_processed
│   └── DEMO998
│       ├── DEMO998_prep1_thumbnail_anchorInitSnakeContours.pkl
│       ├── DEMO998_prep1_thumbnail_initSnakeContours.pkl
```

- Modify `input_spec.ini` as (alignedPadded,NtbNormalized,thumbnail). `python masking.py example_specs/DEMO998_input_spec.ini /home/yuncong/demo_data/CSHL_data_processed/DEMO998/DEMO998_prep1_thumbnail_initSnakeContours.pkl`

```bash
├── CSHL_data_processed
│   └── DEMO998
│       ├── DEMO998_prep1_thumbnail_autoSubmasks
│       │   ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242
│       │   │   ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep1_thumbnail_autoSubmask_0.png
│       │   │   └── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep1_thumbnail_autoSubmaskDecisions.csv
│       │   ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250
│       │   │   ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep1_thumbnail_autoSubmask_0.png
│       │   │   └── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep1_thumbnail_autoSubmaskDecisions.csv
│       │   └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257
│       │       ├── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep1_thumbnail_autoSubmask_0.png
│       │       └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep1_thumbnail_autoSubmaskDecisions.csv
```

- Re-launch masking GUI to inspect, correct the automatically generated masks, then export as PNGs.

```bash
├── CSHL_data_processed
│   └── DEMO998
│       ├── DEMO998_prep1_thumbnail_mask
│       │   ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep1_thumbnail_mask.png
│       │   ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep1_thumbnail_mask.png
│       │   └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep1_thumbnail_mask.png
```

- Modify `input_spec.ini` as (None,NtbNormalized,thumbnail). Run `python generate_original_image_crop_csv.py example_specs/DEMO998_input_spec.ini`. 

```bash
├── CSHL_data_processed
│   └── DEMO998
│       ├── DEMO998_original_image_crop.csv
```
- Copy operation config template. `cp $DATA_ROOTDIR/operation_configs/crop_orig_template.ini $DATA_ROOTDIR/CSHL_data_processed/DEMO998/DEMO998_operation_configs/crop_orig.ini`. Modify `crop_orig.ini`. 
- Modify `input_spec.ini` as (alignedPadded,mask,thumbnail). Run `python warp_crop.py --input_spec example_specs/DEMO998_input_spec.ini --op_id from_padded_to_none`.

```bash
├── CSHL_data_processed
│   └── DEMO998
│       ├── DEMO998_thumbnail_mask
│       │   ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_thumbnail_mask.png
│       │   ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_thumbnail_mask.png
│       │   └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_thumbnail_mask.png
