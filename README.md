# TrueCity: Real and Simulated Urban Data for Cross-Domain 3D Scene Understanding

<p align="center">
  <img src="figures/teaser.jpg" alt="TrueCity Teaser" width="800"/>
</p>

<p align="center">
  <em>TrueCity introduces real-world annotated point clouds, a semantic 3D city model, and 3D-model simulated point clouds of the same location, enabling coherent evaluation of the sim-to-real domain gap in 3D scene understanding.</em>
</p>

## Abstract

3D semantic scene understanding remains a long-standing challenge in the 3D computer vision community. One of the key issues pertains to limited real-world annotated data to facilitate generalizable models. The common practice to tackle this issue is to simulate new data. Although synthetic datasets offer scalability and perfect labels, their designer-crafted scenes fail to capture real-world complexity and sensor noise, resulting in a synthetic-to-real domain gap. Moreover, no benchmark provides synchronized real and simulated point clouds for segmentation-oriented domain shift analysis.

We introduce TrueCity, the first urban semantic segmentation benchmark with cm-accurate annotated real-world point clouds, semantic 3D city models, and annotated simulated point clouds representing the same city. TrueCity proposes segmentation classes aligned with international 3D city modeling standards, enabling consistent evaluation of synthetic-to-real gap. Our extensive experiments on common baselines quantify domain shift and highlight strategies for exploiting synthetic data to enhance real-world 3D scene understanding.

We are convinced that the TrueCity dataset will foster further development of sim-to-real gap quantification and enable generalizable data-driven models.

## TrueCity Benchmark Dataset

<p align="center">
  <img src="figures/benchmark-overview.jpg" alt="Benchmark Overview" width="100%"/>
</p>

<p align="center">
  <em>Real-world point cloud (2nd row), which was manually labeled according to the class list, used for manual modeling of semantic 3D models (3rd row), which in turn were used to simulate and auto-label synthetic point clouds (4th row).</em>
</p>

### 3D Semantic Road Space Classes

TrueCity proposes a class list of **12 classes** harmonized with the standards **CityGML 2.0** and **OpenDRIVE 1.4**:

1. **RoadSurface**: Vehicle-allowed surfaces without sidewalks
2. **GroundSurface**: Pedestrian-allowed surfaces without road surface
3. **CityFurniture**: Vertical urban installation without building-attached objects
4. **Vehicle**: Parked or moving vehicles
5. **Pedestrian**: Standing or moving persons
6. **WallSurface**: Building parts without roofs, installations, facade elements
7. **RoofSurface**: Building parts forming roof structures
8. **Door**: Openings allowing entering objects with gates
9. **Window**: Openings and its outer blinds without entries
10. **BuildingInstallation**: Building-attached installation
11. **SolitaryVegetationObject**: Vegetation with tree trunks and branches
12. **Noise**: Noisy points and any other non-annotated element

Leveraging standardized class definitions facilitates seamless integration and reuse in downstream methods and applications, such as 3D road space and facade semantic reconstruction.


## Experiments

<p align="center">
  <img src="figures/data_split_v2.jpg" alt="Data Split" width="100%"/>
</p>

<p align="center">
  <em>Top-down schematic of S--R mixtures along a continuous streetscape. Solid lines mark train/validation/test splits; dashed lines mark boundaries between contiguous synthetic and real segments for each mixture ratio.</em>
</p>

We evaluate on TrueCity under controlled synthetic-real (%S--%R) mixtures of **100S--0R, 75S--25R, 50S--50R, 25S--75R, and 0S--100R**. We form each S--R mixture by assigning contiguous spatial segments to the synthetic or real domain. The test and validation data remain real-only throughout the experiments.

### Baseline Semantic Segmentation Methods

To probe synthetic-real (S--R) domain shift on TrueCity, we evaluate a representative suite of point-cloud semantic segmentation baselines:

**Point-based:**
- PointNet
- PointNet++
- RandLA-Net

**Kernel-based:**
- KPConv

**Transformer-based:**
- Point Transformer v1
- Point Transformer v3
- Superpoint Transformer
- OctFormer

## Results

The following table shows TrueCity segmentation results under synthetic-real (S--R) mixes. We report mIoU and OA; shifts along S--R reveal family-specific inductive biases in point-, kernel-, and transformer-based models. **Bold** marks the best value for each model across mixtures.

| Model | 100S--0R | | 75S--25R | | 50S--50R | | 25S--75R | | 0S--100R | |
|-------|----------|-------|----------|-------|----------|-------|----------|-------|----------|-------|
| | **mIoU** | **OA** | **mIoU** | **OA** | **mIoU** | **OA** | **mIoU** | **OA** | **mIoU** | **OA** |
| **Point-based** |
| PointNet | 6.03 | 30.36 | 10.74 | 48.10 | 10.89 | 49.29 | 13.10 | 47.99 | **14.51** | **49.82** |
| PointNet++ | 9.72 | 34.39 | 20.95 | 62.80 | 23.18 | **65.36** | **25.38** | 63.27 | 23.39 | 63.15 |
| RandLA-Net | 8.98 | 35.40 | 13.25 | 50.32 | 15.73 | **59.37** | 16.89 | 57.09 | **17.71** | 54.57 |
| **Kernel-based** |
| KPConv | 15.84 | 50.07 | 21.55 | 62.08 | 28.50 | 61.62 | 22.33 | 61.92 | **29.90** | **62.80** |
| **Transformer-based** |
| Point Transformer v1 | 16.30 | 57.54 | 19.79 | 60.29 | 23.43 | 67.54 | 24.66 | **68.70** | **28.89** | 67.98 |
| Point Transformer v3 | 14.13 | 53.15 | 19.29 | 60.22 | **25.30** | **65.94** | 24.64 | 65.72 | 25.24 | 60.75 |
| Superpoint Transformer | 14.31 | 54.17 | 17.01 | **58.62** | 14.22 | 54.63 | 19.61 | **56.98** | 15.96 | 54.64 |
| OctFormer | 13.07 | 53.30 | 14.17 | 55.34 | 14.22 | 49.71 | 13.91 | 50.97 | **17.65** | **56.28** |

<p align="center">
  <img src="figures/experiment_result_final.pdf" alt="Experimental Results" width="100%"/>
</p>

<p align="center">
  <em>Qualitative impact of the synthetic–real (S--R) training mix on models from different methods (Point-based, Kernel-based and Transformer-based). We also present the ground truth synthetic and real point clouds; colors follow the TrueCity legend.</em>
</p>

### Key Findings

**Synthetic data helps, but its utility depends on the model's inductive bias:**

- **Point Transformer v3** improves mIoU from 14.13% at 100S--0R to 25.30% at 50S--50R and 25.24% at 0S--100R
- **Point Transformer v1** benefits primarily from real data, moving from 16.30% at 100S--0R to 28.89% at 0S--100R
- **PointNet++** peaks at 25.38% with 25S--75R versus 23.39% with 0S--100R
- **RandLA-Net** and **Superpoint Transformer** lean heavily on real data

**Insignificant Domain Gap Classes:**
- *WallSurface*, *RoadSurface*, *GroundSurface*, and *SolitaryVegetationObject* show minimal sensitivity to the proportion of real data
- These classes share traits of geometric regularity and relatively simple structural context
- Even small proportions of real data suffice to approach optimal performance

**Significant Domain Gap Classes:**
- *Door*, *RoofSurface*, *BuildingInstallation*, *Noise*, *CityFurniture*, and *Window* display substantial domain gaps
- Performance depends heavily on the balance of synthetic and real supervision
- These classes involve fine-scale geometry, occlusions, or high variability in appearance

## Resources

- **GitHub Repository**: [https://github.com/tum-gis/TrueCity](https://github.com/tum-gis/TrueCity)
- **Data Download**: Available at the GitHub repository
- **3D City Models**: Semantic city models in CityGML 2.0 format
- **OpenDRIVE Dataset**: Road network and roadside objects

## Citation

If you use TrueCity in your research, please cite our work:

```bibtex
@article{truecity2024,
  title={TrueCity: Real and Simulated Urban Data for Cross-Domain 3D Scene Understanding},
  author={Your Authors},
  journal={Your Conference/Journal},
  year={2024}
}
```

## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/tum-gis/TrueCity).

---
