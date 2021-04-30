# Lecture 2 - Creating Training Datasets

## Introduction

The Jupyter Notebook exercises in this lecture will lead you through the process of taking a raw dataset collected in the field and turning it into a machine-learning ready training dataset.
The example dataset used in these exercises is a shapefile which contains geographic bounds of fields in South Africa and various properties including the type of crop grown in that field.
Note that this is an example dataset and the crop types listed are not the actual crop types present.

This lecture is split into four exercises. In the first exercise you will be exploring an input dataset and checking its various properties (attribute values, CRS, temporal range, geographic extent, etc.). In the second exercise you will clean the input dataset. The third exercise will cover matching the input dataset with corresponding Sentinel-2 imagery and downloading the imagery from an S3 bucket. Lastly, the final exercise will cover rasterizing your cleaned label file and cropping the Sentinel-2 imagery to cover only the extent of the label file.

## Python Packages Used

* [**rasterio**](https://rasterio.readthedocs.io/en/latest/) - Used for reading in and writing raster datasets
* [**shapely**](https://shapely.readthedocs.io/en/stable/manual.html) - Used for manipulating geometries and performing spatial queries
* [**fiona**](https://fiona.readthedocs.io/en/latest/) - Reads in all geographic file formats and converts the geometries to shapely geometries
* [**geopandas**](https://geopandas.org/) - Very similar to [pandas](https://pandas.pydata.org/) but also loads geometries into dataframes
* [**arrow**](https://arrow.readthedocs.io/en/latest/) - Simple library for parsing datetime strings
* [**boto3**](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html) - Library for interacting with [AWS](https://aws.amazon.com/) services
* [**pyproj**](https://pyproj4.github.io/pyproj/stable/) - Used for reprojecting geometries from one CRS to another

## Homework

After completing these exercises, try creating a complete training dataset using the same methods with a different input dataset. A good input dataset to use is the label file from the the [Smallholder Cashew Plantations in Benin](https://registry.mlhub.earth/10.34911/rdnt.hfv20i/) dataset hosted on Radiant MLHub.
