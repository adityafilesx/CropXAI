# Vendored third-party code

`prithvi_mae.py` is copied unmodified from the Hugging Face repository
`ibm-nasa-geospatial/Prithvi-EO-1.0-100M`, revision `f3a9ea7a1723621b0aeceb4de993093704d712c5`
(Apache License 2.0, Copyright (c) IBM Corp. 2024; the licence header is retained in the file).
It defines the Prithvi ViT encoder (`PrithviViT`) and MAE decoder. Only the encoder is used by CropXAI.

Pretrained weights are not vendored. They are loaded from
`artifacts/pretrained/prithvi_eo_1.0_100m/Prithvi_EO_V1_100M.pt`
(SHA-256 `7fac0c8a8693198e32a055e0c5a967f8b005f382182b63df1b29fdcd5c880731`) obtained from the same revision.
