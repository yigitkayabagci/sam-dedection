# SegFly figure attribution

segfly-instance-conversion.png derives from one SegFly sample:

- dataset: markus-42/SegFly
- converted parquet: default/train/0212.parquet, row 17
- scene / altitude / modality: scene_03 / 50m / thermal
- source license: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- source project: Markus Gross et al., *SegFly: A 2D-3D-2D Paradigm for
  Aerial RGB-Thermal Semantic Segmentation at Scale*, ECCV 2026

Thermal, registered RGB and semantic annotation panels are SegFly data.
Thing filtering, connected-components/watershed overlays, boxes, labels and
the panel layout were produced by tools/analyze_segfly_instances.py in this
repository. The derived figure is distributed under CC BY-NC-SA 4.0.
