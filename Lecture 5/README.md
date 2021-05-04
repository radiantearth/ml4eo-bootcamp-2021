# Lecture 5 - STAC + Radiant MLHub

## Introduction

The notebooks in this lecture will introduce you to cataloging earth observation (EO) assets using 
the [SpatioTemporal Asset Catalog (STAC)] specification and using the Radiant MLHub API (a
STAC-compliant API) to discover and download ML training datasets. The example data for this lecture
can be found in the [`data`](./data) directory and are the result of working through the [Lecture 
2](../Lecture%202) exercises.

## Exercises

1. [**Create Source Imagery STAC Item**](./1_create_source_imagery_stac_item.ipynb)
   
   Introduces STAC Items and Assets and walks you through the process of creating a STAC Item to 
   catalog our source imagery using the [PySTAC] library.

2. [**Create STAC Label Item**](./2_create_stac_label_item.ipynb)

   Introduces STAC Extensions and the use of the [Label Extension] to describe labeled data for
   training machine learning models, then walks you through constructing an Item that implements the
   Label Extension.

3. [**Create STAC Collection**](./3_create_stac_collection.ipynb)

   Introduces STAC Collections and Catalogs and walks you through the process of constructing a
   Collection containing the Items from the first 2 exercises.

4. [**Training Data from Radiant MLHub**](./4_training_data_from_radiant_mlhub.ipynb)

    Walks through using the [radiant_mlhub] Python client to discover and download training datasets
    from the Radiant MLHub API.
   

## Python Packages Used

* [PySTAC] - Library for working with STAC objects
* [radiant_mlhub] - Client for interacting with the Radiant MLHub API


## Homework

After completing these exercises, try creating a larger Collection using the same methods with a
different input dataset. If you completed the [Lecture 2 Homework] (based on the [Smallholder
Cashew Plantations in Benin](https://registry.mlhub.earth/10.34911/rdnt.hfv20i/) dataset) you could
try cataloging the resources you created as part of that assignment.

[PySTAC]: https://pystac.readthedocs.io/en/latest/
[radiant_mlhub]: https://radiant-mlhub.readthedocs.io/en/latest/
[SpatioTemporal Asset Catalog (STAC)]: https://stacspec.org/
[Label Extension]: https://github.com/stac-extensions/label
[Lecture 2 Homework]: ../Lecture%202/README.md#homework